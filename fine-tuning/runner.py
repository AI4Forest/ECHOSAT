import os
import platform
import re
import shutil
import sys
import time
from typing import Optional, Any

import numpy as np
import torch
import wandb
from torch.cuda.amp import autocast
from torchmetrics import MeanMetric
from tqdm.auto import tqdm

# Conditional import moved to get_model method

import visualization
from datasetClass import SatelliteImageDataset4D
from metrics import MetricsClass
from utilities import SequentialSchedulers

from models.swin_video_unet import SwinVideoUnet

class Runner:
    """Base class for all runners, defines the general functions"""

    def __init__(self, config: Any, tmp_dir: str, debug: bool):
        """
        Initialize useful variables using config.
        :param config: wandb run config
        :type config: wandb.config.Config
        :param debug: Whether we are in debug mode or not
        :type debug: bool
        """
        self.config = config
        self.debug = debug

        # Set the device
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"Number of GPUs available: {n_gpus}")
            config.update(dict(device='cuda:0'))
        else:
            print("No GPUs available.")
            config.update(dict(device='cpu'))

        self.dataParallel = (torch.cuda.device_count() > 1)
        if not self.dataParallel:
            self.device = torch.device(config.device)
            if 'gpu' in config.device:
                torch.cuda.set_device(self.device)
        else:
            # Use all visible GPUs
            self.device = torch.device("cuda:0")
            torch.cuda.device(self.device)
        torch.backends.cudnn.benchmark = True

        # Set a couple useful variables
        self.seed = int(self.config.seed)
        self.loss_name = self.config.loss_name or 'shift_l1'
        sys.stdout.write(f"Using loss: {self.loss_name}.\n")
        self.use_amp = self.config.fp16
        self.tmp_dir = tmp_dir
        print(f"Using temporary directory {self.tmp_dir}.")

        # Variables to be set
        self.loader = {loader_type: None for loader_type in ['train', 'val']}
        self.loss_criteria = {loss_name: self.get_loss(loss_name=loss_name) for loss_name in ['l1', 'l2', 'huber', 'combi', 'regression', 'disturbance_regression', self.loss_name]}
        for threshold in [15, 20, 25, 30]:
            self.loss_criteria[f"l1_{threshold}"] = self.get_loss(loss_name=f"l1", threshold=threshold)

        if self.config.loss_name == 'combi_2heads':
            self.loss_criteria['l1_head0'] = self.get_loss(loss_name='l1_head0')

        if self.config.use_overlapping_patches:
            self.loss_criteria['overlap'] = self.get_loss(loss_name='overlap')

        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.artifact = None
        self.model_paths = {model_type: None for model_type in ['initial', 'trained']}
        if self.config.get("model_checkpoint", None) is not None:
            self.model_paths['initial'] = self.config.model_checkpoint
        self.model_metrics = {  # Organized way of saving metrics needed for retraining etc.
        }

        self.metrics = {mode: {'loss': MeanMetric().to(device=self.device),
                               'l1': MeanMetric().to(device=self.device),
                               'l2': MeanMetric().to(device=self.device),
                               'huber': MeanMetric().to(device=self.device),
                               'combi': MeanMetric().to(device=self.device),
                               'regression': MeanMetric().to(device=self.device),
                               'disturbance_regression': MeanMetric().to(device=self.device),
                               self.config.loss_name: MeanMetric().to(device=self.device)
                               }
                        for mode in ['train', 'val']}

        if self.config.use_overlapping_patches:
            self.metrics['train']['overlap'] = MeanMetric().to(device=self.device)
            self.metrics['val']['overlap'] = MeanMetric().to(device=self.device)

        if self.config.loss_name == 'combi_2heads':
            self.metrics['train']['l1_head0'] = MeanMetric().to(device=self.device)
            self.metrics['val']['l1_head0'] = MeanMetric().to(device=self.device)    

        for mode in ['train', 'val']:
            for threshold in [15, 20, 25, 30]:
                self.metrics[mode][f"l1_{threshold}"] = MeanMetric().to(device=self.device)
        
        self.metrics['train']['ips_throughput'] = MeanMetric().to(device=self.device)

        self.use_early_stopping = config.early_stopping and (not self.debug)
        self.best_val_loss = float('inf')
        self.metrics['best_val'] = {metric_name: MeanMetric().to(device=self.device) for metric_name in self.metrics['val'].keys()}
        self.best_model_path = None  # Path to save the best model

    @staticmethod
    def set_seed(seed: int):
        """
        Sets the seed for the current run.
        :param seed: seed to be used
        """
        # Set a unique random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(seed)  # This works if CUDA not available

    def reset_averaged_metrics(self):
        """Resets all metrics"""
        for mode in self.metrics.keys():
            for metric in self.metrics[mode].values():
                metric.reset()

    def get_metrics(self) -> dict:
        """
        Returns the metrics for the current epoch.
        :return: dict containing the metrics
        :rtype: dict
        """
        with torch.no_grad():
            loggingDict = dict(
                # Model metrics
                n_params=MetricsClass.get_parameter_count(model=self.model),

                # Optimizer metrics
                learning_rate=float(self.optimizer.param_groups[0]['lr']),
            )
            # Add metrics
            for split in ['train', 'val']:
                for metric_name, metric in self.metrics[split].items():
                    try:
                        # Catch case where MeanMetric mode not set yet
                        loggingDict[f"{split}/{metric_name}"] = metric.compute()
                    except Exception as e:
                        continue

        return loggingDict

    @staticmethod
    def get_dataset_root(dataset_name: str) -> str: # TODO: this is not yet adapted to the new caching system
        """Returns the dataset rootpath for the given dataset.
        If we are on a z1 node and the dataset is not cached locally, it will be copied to the local cache."""
        sys.stdout.write(f"Loading {dataset_name}.\n")
        is_htc = 'htc-' in platform.uname().node
        is_coder = 'coder-' in platform.uname().node
        is_gcp = 'gcp-' in platform.uname().node
        is_palma = bool(re.match(r'r\d+n\d+', platform.uname().node))

        # Determine where the data lies
        for permanent_cache in ['.']:  # AIS2T Permanent Storage, AIS2T SCRATCH, scratch_jan, palma_jan
            permanent_dataset_root = os.path.join(permanent_cache, dataset_name)
            if os.path.isdir(permanent_dataset_root):
                print(f"Found dataset in {permanent_cache}.")
                print(f"Using dataset root: {permanent_dataset_root}.")
                break

        #local_cache = '/scratch/local/ais2t_vision_cache'
        local_cache = '/scratch/local'
        local_dataset_root = os.path.join(local_cache, dataset_name)

        dataset_is_permanently_cached = os.path.exists(permanent_dataset_root)
        #assert is_htc or is_gcp or is_coder or is_palma, "This function is only intended to be used on htc or coder nodes and currently only allows already downloaded datasets."
        # assert dataset_is_permanently_cached, "The dataset is not permanently cached on z1, aborting."
        if is_gcp:
            assert permanent_cache == '/home/htc/mzimmer/SCRATCH', "GCP needs datasets on GCP storage."

        is_copyable = dataset_name in ['ai4forest_6_12_256_256', 'ai4forest_same_s1_s2_6_12_256_256', 'ai4forest_random_s1_s2_6_12_256_256', 'ai4forest_2020_12_12_256_256']
        if is_htc and is_copyable:
            # We are running on a z1 node -> check whether the dataset is cached locally, otherwise cache it locally from the permanent cache
            busyFile = os.path.join(local_cache, f"{dataset_name}-busyfile.lock")
            doneFile = os.path.join(local_cache, f"{dataset_name}-donefile.lock")
            if os.path.exists(local_dataset_root) and os.path.exists(doneFile):
                # The dataset is cached locally on the node and we can use it
                dataset_root = local_dataset_root
            else:
                # The dataset is not cached locally, hence we need to copy it
                os.makedirs(local_cache, exist_ok=True) # Create the cache directory if it does not exist

                wait_it = 0
                while True:
                    is_done = os.path.exists(doneFile)
                    is_busy = os.path.exists(busyFile)
                    if is_done:
                        # Dataset exists locally
                        dataset_root = local_dataset_root
                        sys.stdout.write("Local data storage: Done file exists.\n")
                        break
                    elif is_busy:
                        # Wait for 10 seconds, then check again
                        time.sleep(10)
                        sys.stdout.write("Local data storage: Is still busy - wait.\n")
                        continue
                    else:
                        # Create the busyFile
                        open(busyFile, mode='a').close()

                        # Copy the dataset
                        sys.stdout.write(f"Local data storage: Starts copying from {permanent_dataset_root} to {local_dataset_root}.\n")
                        try:
                            shutil.copytree(src=permanent_dataset_root, dst=local_dataset_root)
                        except Exception as e:
                            sys.stdout.write(f"Exception raised, continuing in while loop. Exception: {e}")
                            time.sleep(10)
                            continue
                        sys.stdout.write("Local data storage: Copying done.\n")
                        # Create the doneFile
                        open(doneFile, mode='a').close()

                        # Remove the busyFile
                        os.remove(busyFile)

                    wait_it += 1
                    if wait_it == 360:
                        # Waited 1 hour, this should be done by now, check for errors
                        raise Exception("Waiting time too long. Check for errors.")

        elif is_coder:
            # We are running on a coder node and hence load the dataset directly from the permanent cache on z1
            dataset_root = os.path.join(permanent_cache, dataset_name)
        else:
            # Use the permanent dataset without copying
            dataset_root = permanent_dataset_root
        return dataset_root

    def get_dataloaders(self):
        base_dataset_name = self.config.dataset
        rootPath = self.get_dataset_root(dataset_name=base_dataset_name)
        print(f"Loading {base_dataset_name} dataset from {rootPath}.")

        trainData = SatelliteImageDataset4D(data_path=rootPath, use_reduced_input_size=self.config.use_reduced_input_size, use_overlapping_patches=self.config.use_overlapping_patches, overlap_size=self.config.overlap_size)
        fixed_val_data = SatelliteImageDataset4D(data_path=rootPath, fixed_val=True, use_reduced_input_size=self.config.use_reduced_input_size)

        # Select at maximum 10% of the data for validation using a fixed random seed. The total amount should be capped as 300
        cut_off = 300
        if self.debug:
            cut_off = int(0.1 * cut_off)
        n_val = min(int(0.1 * len(trainData)), cut_off)
        n_train = len(trainData) - n_val
        trainData, valData = torch.utils.data.random_split(trainData, [n_train, n_val], generator=torch.Generator().manual_seed(1234))

        sys.stdout.write(f"Length of train and val splits: {len(trainData)}, {len(valData)}.\n")

        num_workers_default = self.config.num_workers_per_gpu if self.config.num_workers_per_gpu is not None else 8
        num_workers = num_workers_default * torch.cuda.device_count() * int(not self.debug) 
        prefetch_factor = self.config.prefetch_factor if self.config.prefetch_factor is not None else 1

        sys.stdout.write(f"Using {num_workers} workers.\n")
        train_sampler = None
        shuffle = True
        if self.debug:
            prefetch_factor = None
            
        if self.config.use_overlapping_patches:
            assert self.config.batch_size % 2 == 0, f"Batch size must be even if using overlapping patches, not {self.config.batch_size}."
            batch_size = self.config.batch_size // 2
            
            def collate_fn(x):
                inputs, labels = zip(*x)
                return torch.stack([item for tuplee in inputs for item in tuplee]), torch.stack([item for tuplee in labels for item in tuplee])
            
            collate_fn = collate_fn
        else:
            batch_size = self.config.batch_size
            collate_fn = None
        
        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, sampler=train_sampler,
                                                  pin_memory=torch.cuda.is_available(), num_workers=num_workers,
                                                  shuffle=shuffle, prefetch_factor=prefetch_factor, collate_fn=collate_fn)
        
        # Only use deterministic samples for validation metrics and more importantly images
        generator = torch.Generator().manual_seed(42)
        valLoader = torch.utils.data.DataLoader(valData, batch_size=batch_size, shuffle=False, generator=generator,
                                                pin_memory=torch.cuda.is_available(), num_workers=num_workers, prefetch_factor=prefetch_factor, collate_fn=collate_fn)
        
        fixedValLoader = torch.utils.data.DataLoader(fixed_val_data, batch_size=1, shuffle=False, generator=generator,
                                                pin_memory=torch.cuda.is_available(), num_workers=num_workers, prefetch_factor=prefetch_factor)

        return trainLoader, valLoader, fixedValLoader

    def get_model(self, reinit: bool, model_path: Optional[str] = None) -> torch.nn.Module:
        """
        Returns the model.
        :param reinit: If True, the model is reinitialized.
        :param model_path: Path to the model.
        :return: The model.
        """
        print(f"Loading model - reinit: {reinit} | path: {model_path if model_path else 'None specified'}.")
        if reinit:
            # Define the model architecture
            arch = self.config.arch or 'unet3d'
            
            # The dataset naming convention ends with _years_months_channels_width_height, e.g. _4_6_12_256_256
            # For 3D UNets we use the number of months to compute input channels,
            # but for our hierarchical (multi-year) network we assume the input has shape:
            # [batch, n_bands, years, months, H, W]
            if arch == 'unet3d':
                # For 3D models
                years, months, _, _, _ = self.config.dataset.split('_')[-5:]
                months = int(months)
                sys.stdout.write(f"Using {months} months.\n")
                assert months in [6, 12], f"Only 6 and 12 months are supported, not {months}."
                n_bands = 14
                

                if months == 6:
                    model = SingleYearUNetSixMonth(n_channels=n_bands)
                elif months == 12:
                    model = SingleYearUNetTwelveMonth(n_channels=n_bands)

            elif arch == 'unet4d':
                # For the hierarchical (4D) model, the input tensor is expected to have shape:
                # [batch, n_bands, years, months, H, W]
                # Here, we do not multiply n_bands by months.
                n_bands = 14
                in_channels = n_bands
                
                aggregation = self.config.aggregation if 'aggregation' in self.config.keys() else "kernel7"
                sys.stdout.write(f"Using hierarchical 4D network with input channels = {in_channels}.\n")
                # Instantiate your hierarchical 4D UNet. Adjust month_feature_dim and final_dims as needed.
                model = HierarchicalUNet4D(n_channels=in_channels, month_feature_dim=64, final_dims=1, aggregation=aggregation)
            elif arch == "unet4d_mini":
                
                n_bands = 14
                in_channels = n_bands
                sys.stdout.write(f"Using hierarchical mini 4D network with input channels = {in_channels}.\n")
                model = HierarchicalUNet4D(n_channels=in_channels, month_feature_dim=16, final_dims=1, mini=True)
            elif arch == 'swin_transformer':
                from models.swin_transformer import SwinTransformerWithFullUnetDecoder
                model = SwinTransformerWithFullUnetDecoder()
            elif arch == 'swin_video_unet':
                print("Using SwinVideoUnet architecture.")
                # Determine input spatial size based on configuration
                spatial_size = self.config.use_reduced_input_size if self.config.use_reduced_input_size > 0 else 256
                downsample_per_year = getattr(self.config, 'downsample_per_year', False)
                
                # Get patch sizes from config, with fallback defaults
                patch_size_time = getattr(self.config, 'patch_size_time', 2)
                patch_size_image = getattr(self.config, 'patch_size_image', 4)
                reduce_time = getattr(self.config, 'reduce_time', (16, 8, 4))
                window_size_temporal = getattr(self.config, 'window_size_temporal', 2)
                window_size_spatial = getattr(self.config, 'window_size_spatial', 4)
                encoder_depths = getattr(self.config, 'encoder_depths', (6, 4, 4, 2))
                decoder_depths = getattr(self.config, 'decoder_depths', (2, 4, 4, 4))
                embed_dim = getattr(self.config, 'embed_dim', 72)
                temporal_skip_reduction = getattr(self.config, 'temporal_skip_reduction', "linear")
                temporal_skip_reduction = "linear" if temporal_skip_reduction is None else temporal_skip_reduction
                use_final_convs = getattr(self.config, 'use_final_convs', False)
                
                
                # Override patch_size_image to 1 if using reduced input size and default patch size
                if self.config.use_reduced_input_size and patch_size_image == 4:
                    print("WARNING: Using patch size 4 along with reduced input size...")
                
                print(f"Using patch sizes: time={patch_size_time}, image={patch_size_image}")
                print(f"Using encoder depths: {encoder_depths}, decoder depths: {decoder_depths}")
                
                model = SwinVideoUnet(
                    # 4 Years, 12 Months, 18 Channels, Spatial_size, Spatial_size
                    input_shape=(7, 12, 18, spatial_size, spatial_size),
                    embed_dim=embed_dim,
                    encoder_depths=encoder_depths,
                    decoder_depths=decoder_depths,
                    num_heads=(4, 8, 12, 24),
                    window_size_temporal=window_size_temporal,
                    window_size_spatial=window_size_spatial,
                    reduce_time=reduce_time,
                    patch_size_time=patch_size_time,
                    patch_size_image=patch_size_image,
                    temporal_skip_reduction=temporal_skip_reduction,
                    use_final_convs=use_final_convs,
                    downsample_per_year=downsample_per_year
                )

            else:
                raise ValueError(f"Unknown architecture: {arch}")

        else:
            # Use the already initialized model
            model = self.model

        if model_path is not None:
            new_loading = '.ckpt' in model_path
            if new_loading:
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint['state_dict']
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
            else:    
                state_dict = torch.load(model_path, map_location=self.device)
            new_state_dict = {}
            require_DP_format = isinstance(model, torch.nn.DataParallel)
            for k, v in state_dict.items():
                is_in_DP_format = k.startswith("module.")
                if require_DP_format and is_in_DP_format:
                    name = k
                elif require_DP_format and not is_in_DP_format:
                    name = "module." + k
                elif not require_DP_format and is_in_DP_format:
                    name = k[7:]
                else:
                    name = k
                new_state_dict[name] = v
            try:
                model.load_state_dict(new_state_dict, strict=True)
            except Exception as e:
                print(f"Error loading model: {e}")
                # Try loading with strict=False to get missing/unexpected keys
                incompatible = model.load_state_dict(new_state_dict, strict=False)
                if hasattr(incompatible, 'missing_keys') and hasattr(incompatible, 'unexpected_keys'):
                    if incompatible.missing_keys:
                        print(f"Missing keys: {incompatible.missing_keys}")
                    if incompatible.unexpected_keys:
                        print(f"Unexpected keys: {incompatible.unexpected_keys}")
                else:
                    # Fallback for older torch versions
                    missing_keys, unexpected_keys = incompatible
                    if missing_keys:
                        print(f"Missing keys: {missing_keys}")
                    if unexpected_keys:
                        print(f"Unexpected keys: {unexpected_keys}")
                print(f"Loading model with strict=False")

        if self.config.loss_name == 'combi_2heads' and reinit:
            out_channels = 2
            import torch.nn as nn
            try:
                model = model.module
            except Exception as e:
                pass
            
            freeze_model = self.config.get('freeze_model', 'None')
            if freeze_model in ['full', 'model_only']:
                for param in model.parameters():
                    param.requires_grad = False
            # for param in model.final_conv3.parameters():
            #     param.requires_grad = False        
            # weight = model.final_conv3.weight.data
            # bias = model.final_conv3.bias.data
            # model.final_conv3 = nn.Conv3d(48, out_channels, kernel_size=(1, 1, 1))
            # for param in model.final_conv3_2.parameters():
            #     param.requires_grad = False
            # model.final_conv3.weight[0:1] = weight.detach().clone()
            # model.final_conv3.bias[0:1] = bias.detach().clone()
            # model.final_conv3_2.weight[:] = weight.clone()
            # model.final_conv3_2.bias[:] = bias.detach().clone()
            for param in model.final_conv3_2.parameters():
                param.requires_grad = True
            if freeze_model == 'model_only':    
                for param in model.final_conv3.parameters():
                    param.requires_grad = True   
        if self.dataParallel and reinit and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)
        model = model.to(device=self.device)
        return model

    def get_loss(self, loss_name: str, threshold: float = None):
        assert loss_name in ['shift_l1', 'shift_l2', 'shift_huber', 'l1', 'l2', 'huber', 'growth', 'overlap', 'combi', 'regression', 'disturbance_regression', 'combi_2heads', 'l1_head0'], f"Loss {loss_name} not implemented."
        if threshold is not None:
            assert loss_name == 'l1', f"Threshold only implemented for l1 loss, not {loss_name}."
        # Dim 1 is the channel dimension, 0 is batch.
        # Sums up to get average height, could be mean without zeros
        expand_labels_over_time = self.config.expand_labels_over_time if 'expand_labels_over_time' in self.config.keys() else False
        if self.config.loss_name == 'combi_2heads':
            if loss_name != 'combi_2heads':
                remove_sub_track = lambda out, target: (out[:,1,:,:,:], torch.sum(target, dim=-3))
            else:
                remove_sub_track = lambda out, target: (out, torch.sum(target, dim=-3))
        else:    
            remove_sub_track = lambda out, target: (out, torch.sum(target, dim=-3))
        def remove_sub_track_expand_labels_in_time(out, target):
            # Sums up the labels over the time dimension to get the total canopy height
            target = target.sum(dim=-3) ## remove subtrack by summing over granule
            # target = target.max(dim=-3, keepdim=True).values ## take maximal height over time
            target = target[:,2:3,:,:]
            target = target.expand(out.shape) ## expand maximal height to all time steps
            return out, target
        if expand_labels_over_time:
            pre_calculation_function = remove_sub_track_expand_labels_in_time
        else:
            pre_calculation_function = remove_sub_track  
        if loss_name == 'shift_l1':
            from losses.shift_l1_loss import ShiftL1Loss
            loss = ShiftL1Loss(ignore_value=0)
        elif loss_name == 'shift_l2':
            from losses.shift_l2_loss import ShiftL2Loss
            loss = ShiftL2Loss(ignore_value=0)
        elif loss_name == 'shift_huber':
            from losses.shift_huber_loss import ShiftHuberLoss
            loss = ShiftHuberLoss(ignore_value=0)
        elif loss_name == 'l1':
            from losses.l1_loss import L1Loss
            # Rescale the threshold to account for the label rescaling
            if threshold is not None:
                threshold = threshold      
            loss = L1Loss(ignore_value=0, pre_calculation_function=pre_calculation_function, lower_threshold=threshold)
        elif loss_name == 'l2':
            from losses.l2_loss import L2Loss
            loss = L2Loss(ignore_value=0, pre_calculation_function=pre_calculation_function)
        elif loss_name == 'huber':
            from losses.huber_loss import HuberLoss
            loss = HuberLoss(ignore_value=0, pre_calculation_function=pre_calculation_function, delta=3.0)
        elif loss_name == 'combi':
            from losses.combi_loss import CombiLoss
            lambda_regression = self.config.lambda_regression if 'lambda_regression' in self.config.keys() else 1.0
            full_disturbance_window = self.config.full_disturbance_window if 'full_disturbance_window' in self.config.keys() else True
            disturbance_indicator = self.config.get('disturbance_indicator', 7)
            slope_min = self.config.get('slope_min', 0)
            slope_max = self.config.get('slope_max', 2)
            loss = CombiLoss(ignore_value=0, pre_calculation_function=pre_calculation_function, 
                             lambda_regression=lambda_regression, full_disturbance_window=full_disturbance_window,
                             disturbance_indicator=disturbance_indicator, slope_min=slope_min, slope_max=slope_max)
        elif loss_name == 'regression':
            from losses.regression_loss import RegressionLoss
            full_disturbance_window = self.config.full_disturbance_window if 'full_disturbance_window' in self.config.keys() else True
            loss = RegressionLoss(ignore_value=0, pre_calculation_function=pre_calculation_function, full_disturbance_window=full_disturbance_window)
        elif loss_name == 'disturbance_regression':
            from losses.regression_loss_disturbance import DisturbanceRegressionLoss
            full_disturbance_window = self.config.full_disturbance_window if 'full_disturbance_window' in self.config.keys() else True
            use_l2 = self.config.use_l2 if 'use_l2' in self.config.keys() else True
            disturbance_indicator = self.config.get('disturbance_indicator', 7)
            slope_min = self.config.get('slope_min', 0)
            slope_max = self.config.get('slope_max', 2)
            loss = DisturbanceRegressionLoss(disturbance_indicator=disturbance_indicator, slope_min=slope_min, slope_max=slope_max, 
                                             full_disturbance_window=full_disturbance_window, 
                                             use_l2=use_l2, precalculation_function=pre_calculation_function)
        elif loss_name == 'overlap':
            from losses.overlap_loss import OverlapLoss
            overlap_lambda = self.config.overlap_lambda if 'overlap_lambda' in self.config.keys() else 1.0
            overlap_size = self.config.overlap_size if 'overlap_size' in self.config.keys() else 40
            loss = OverlapLoss(ignore_value=0, pre_calculation_function=pre_calculation_function, overlap_lambda=overlap_lambda, overlap_size=overlap_size)
        elif loss_name == 'combi_2heads':
            from losses.combi_loss_2heads import CombiLoss2Heads
            lambda_regression = self.config.lambda_regression if 'lambda_regression' in self.config.keys() else 1.0
            full_disturbance_window = self.config.full_disturbance_window if 'full_disturbance_window' in self.config.keys() else True
            disturbance_indicator = self.config.get('disturbance_indicator', 7)
            slope_min = self.config.get('slope_min', 0)
            slope_max = self.config.get('slope_max', 2)
            max_intercept_after_disturbance = self.config.get('max_intercept_after_disturbance', 100.0)
            disturbance_factor = self.config.get('disturbance_factor', 1.0)
            no_disturbance_factor = self.config.get('no_disturbance_factor', 1.0)
            slope_no_disturbance = self.config.get('slope_no_disturbance', None)
            loss = CombiLoss2Heads(ignore_value=0, pre_calculation_function=pre_calculation_function, 
                                   disturbance_indicator=disturbance_indicator, slope_min=slope_min, slope_max=slope_max,
                                   lambda_regression=lambda_regression, full_disturbance_window=full_disturbance_window,
                                   max_intercept_after_disturbance=max_intercept_after_disturbance,
                                   disturbance_factor=disturbance_factor,
                                   no_disturbance_factor=no_disturbance_factor,
                                   slope_no_disturbance=slope_no_disturbance)
        elif loss_name == 'l1_head0':
            from losses.l1_loss import L1Loss
            def head0_pre_calculation_function(out, target):
                if out.ndim > 1 and out.shape[1] == 2:
                    out = out[:,0,...]
                target = target.sum(dim=-3) ## remove subtrack by summing over granule
                return out, target
            loss = L1Loss(ignore_value=0, pre_calculation_function=head0_pre_calculation_function)   
        loss = loss.to(device=self.device)
        return loss

    def get_visualization(self, viz_name: str, inputs, labels, outputs):
        assert viz_name in ['input_output',
                            'boxplot', 'multi_year_evolution'], f"Visualization {viz_name} not implemented."

        # Detach and copy the labels and outputs, then undo the rescaling
        labels, outputs = labels.detach().clone(), outputs.detach().clone()


        def remove_sub_track_vis(inputs, labels, outputs):
            if outputs.ndim > 1:
                if self.config.loss_name in ['combi_2heads'] and outputs.ndim >= 4:
                    # Gaussian NLL case
                    return  (inputs + 1) / 2 if outputs.ndim > 1 else inputs, labels.sum(
                        axis=-3), outputs[:,1,...]  # Same as remove_sub_track, but for visualization (i.e. has outputs as well)
                
                return  (inputs + 1) / 2 if outputs.ndim > 1 else inputs, labels.sum(
                    axis=-3), outputs  # Same as remove_sub_track, but for visualization (i.e. has outputs as well)
            return  (inputs + 1) / 2 if outputs.ndim > 1 else inputs, labels, outputs
        
        if viz_name == 'input_output':
            viz_fn = visualization.get_input_output_visualization(rgb_channels=[3,2,1],
                                                                      process_variables=remove_sub_track_vis)
        elif viz_name == 'density_scatter_plot':
            viz_fn = visualization.get_density_scatter_plot_visualization(ignore_value=0,
                                                                          process_variables=remove_sub_track_vis)
        elif viz_name == 'boxplot':
            viz_fn = visualization.get_visualization_boxplots(ignore_value=0, process_variables=remove_sub_track_vis)
        elif viz_name == 'multi_year_evolution':
            viz_fn = visualization.get_multi_year_evolution_visualization(process_variables=remove_sub_track_vis)
        return viz_fn(inputs=inputs, labels=labels, outputs=outputs)

    def get_optimizer(self, initial_lr: float) -> torch.optim.Optimizer:
        """
        Returns the optimizer.
        :param initial_lr: The initial learning rate
        :type initial_lr: float
        :return: The optimizer.
        :rtype: torch.optim.Optimizer
        """
        wd = self.config['weight_decay'] or 0.
        optim_name = self.config.optim or 'AdamW'
        if optim_name == 'SGD':
            optimizer = torch.optim.SGD(params=self.model.parameters(), lr=initial_lr,
                                        momentum=0.9,
                                        weight_decay=wd,
                                        nesterov=wd > 0.)
        elif optim_name == 'AdamW':
            optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=initial_lr,
                                          weight_decay=wd)
        else:
            raise NotImplementedError

        return optimizer

    def save_model(self, model_identifier: str, sync: bool = False) -> str:
        """
        Saves the model's state_dict to a file.
        :param model_identifier: Name of the file type.
        :type model_identifier: str
        :param sync: Whether to sync the file to wandb.
        :type sync: bool
        :return: Path to the saved model.
        :rtype: str
        """
        fName = f"{model_identifier}_model.pt"
        fPath = os.path.join(self.tmp_dir, fName)

        # Only save models in their non-module version, to avoid problems when loading
        try:
            model_state_dict = self.model.module.state_dict()
        except AttributeError:
            model_state_dict = self.model.state_dict()

        torch.save(model_state_dict, fPath)  # Save the state_dict

        if sync and not self.debug:
            wandb.save(fPath)
        return fPath

    def log(self, step: int, phase_runtime: float, commit: bool = True):
        """
        Logs the current training status.
        :param phase_runtime: The wall-clock time of the current phase.
        :type phase_runtime: float
        """
        loggingDict = self.get_metrics()
        loggingDict.update({
            'phase_runtime': phase_runtime,
            'iteration': step,
            'samples_seen': step * self.config.batch_size,
        })

        # Log and push to Wandb
        for metric_type, val in loggingDict.items():
            wandb.run.summary[f"{metric_type}"] = val

        wandb.log(loggingDict, commit=commit)
    
    def log_initial_state(self):
        # Log the initial state of the model before any training iteration
        phase_runtime = 0  # No runtime yet
        step = 0  # Initial step
        x_input, y_target = next(iter(self.loader['train']))
        x_input = x_input.to(device=self.device, non_blocking=True)
        y_target = y_target.to(device=self.device, non_blocking=True)
        output = self.model.eval()(x_input)

        # Create the visualizations
        for viz_func in ['input_output', 'boxplot', 'multi_year_evolution']:
            viz = self.get_visualization(viz_name=viz_func, inputs=x_input, labels=y_target, outputs=output)
            wandb.log({'train/' + viz_func: wandb.Image(viz)}, commit=False)

        if not self.debug:
            # Evaluate the validation dataset
            self.eval(data='val')

        self.log(step=step, phase_runtime=phase_runtime)
        self.reset_averaged_metrics()

    def define_optimizer_scheduler(self):
        # Define the optimizer
        initial_lr = float(self.config.initial_lr)
        if self.config.get('freeze_model', 'None') in ['full', 'model_only']:
            initial_lr = float(self.config.initial_lr) * 30
        self.optimizer = self.get_optimizer(initial_lr=initial_lr)

        # We define a scheduler. All schedulers work on a per-iteration basis
        n_total_iterations = self.config.n_iterations
        n_warmup_iterations = int(0.1 * n_total_iterations)

        # Set the initial learning rate
        for param_group in self.optimizer.param_groups: param_group['lr'] = initial_lr

        # Define the warmup scheduler if needed
        warmup_scheduler, milestone = None, None
        if n_warmup_iterations > 0:
            # As a start factor we use 1e-20, to avoid division by zero when putting 0.
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                                 start_factor=1e-20, end_factor=1.,
                                                                 total_iters=n_warmup_iterations)
            milestone = n_warmup_iterations + 1

        n_remaining_iterations = n_total_iterations - n_warmup_iterations
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                      start_factor=1.0, end_factor=0.,
                                                      total_iters=n_remaining_iterations)

        # Reset base lrs to make this work
        scheduler.base_lrs = [initial_lr if warmup_scheduler else 0. for _ in self.optimizer.param_groups]

        # Define the Sequential Scheduler
        if warmup_scheduler is None:
            self.scheduler = scheduler
        else:
            self.scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[warmup_scheduler, scheduler],
                                                  milestones=[milestone])

    @torch.no_grad()
    def eval(self, data: str):
        """
        Evaluates the model on the given data set.
        :param data: string indicating the data set to evaluate on. Can be 'train', 'val', or 'best_val'.
        :type data: str
        """
        if data == 'best_val':
            sys.stdout.write(f"Evaluating early stopping model on fixed validation split.\n")
            dataloader_id = 'fixed_val'
        else:
            sys.stdout.write(f"Evaluating on {data} split.\n")
            dataloader_id = data
        for step, batch in enumerate(tqdm(self.loader[dataloader_id]), 1):
            if len(batch) == 2:
                x_input, y_target = batch
            else:
                x_input, y_target, name = batch
            x_input = x_input.to(self.device, non_blocking=True)
            y_target = y_target.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                output = self.model.eval()(x_input)
                loss = self.loss_criteria[self.loss_name](output, y_target)
                self.metrics[data]['loss'](value=loss, weight=len(y_target))
                for loss_type in self.loss_criteria.keys():
                    metric_loss = self.loss_criteria[loss_type](output, y_target)
                    # Check if the metric_loss is nan
                    if not torch.isnan(metric_loss):
                        self.metrics[data][loss_type](value=metric_loss, weight=len(y_target))
            limit = 10 if data == 'best_val' else 4
            if step <= limit:
                # Create the visualizations for the first 4 batches
                viz_funcs = ['multi_year_evolution'] if data == 'best_val' else ['input_output', 'boxplot', 'multi_year_evolution']
                for viz_func in viz_funcs:
                    viz = self.get_visualization(viz_name=viz_func, inputs=x_input, labels=y_target, outputs=output)
                    if data == 'best_val':
                        wandb.log({data + '/' + viz_func + "_" + str(step) + "_" + str(name[0])[2:-1]: wandb.Image(viz)}, commit=False)
                    else:
                        wandb.log({data + '/' + viz_func + "_" + str(step): wandb.Image(viz)}, commit=False)
            torch.cuda.empty_cache()    # Might help with memory issues according to this thread: https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354


    def train(self):
        log_freq, n_iterations = self.config.log_freq, self.config.n_iterations
        ampGradScaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.reset_averaged_metrics()
        phase_start = time.time()

        # Determine log steps
        if self.debug:
            log_steps = []
        elif isinstance(log_freq, float) and 0 < log_freq < 1:
            log_steps = set(int(i * n_iterations * log_freq) for i in range(1, int(1 / log_freq) + 1))
        else:
            log_steps = set(range(log_freq, n_iterations + 1, log_freq))

        # Initial logging before any training iteration
        # if not self.debug:
        #     self.log_initial_state()

        for step in tqdm(range(1, n_iterations + 1, 1)):
            # Reinitialize the train iterator if it reaches the end
            if step == 1 or (step - 1) % len(self.loader['train']) == 0:
                train_iterator = iter(self.loader['train'])

            # Move to CUDA if possible
            batch = next(train_iterator)
            x_input, y_target = batch
            x_input = x_input.to(device=self.device, non_blocking=True)
            y_target = y_target.to(device=self.device, non_blocking=True)

            self.optimizer.zero_grad()

            itStartTime = time.time()
            with autocast(enabled=self.use_amp):
                output = self.model.train()(x_input)
                loss = self.loss_criteria[self.loss_name](output, y_target)
                ampGradScaler.scale(loss).backward()  # Scaling + Backpropagation
                # Unscale the weights manually, normally this would be done by ampGradScaler.step(), but since
                # we might use gradient clipping, this has to be split
                ampGradScaler.unscale_(self.optimizer)
                if self.config.use_grad_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                ampGradScaler.step(self.optimizer)
                ampGradScaler.update()  # This should happen only once per iteration
                self.scheduler.step()
                self.metrics['train']['loss'](value=loss, weight=len(y_target))
                
                with torch.no_grad():
                    for loss_type in self.loss_criteria.keys():
                        metric_loss = self.loss_criteria[loss_type](output, y_target)
                        # Check if the metric_loss is nan
                        if not torch.isnan(metric_loss):
                            self.metrics['train'][loss_type](value=metric_loss, weight=len(y_target))
                itEndTime = time.time()
                n_img_in_iteration = int(self.config.batch_size)
                ips = n_img_in_iteration / (itEndTime - itStartTime)  # Images processed per second
                self.metrics['train']['ips_throughput'](ips)

            if step in log_steps or step == n_iterations:
                phase_runtime = time.time() - phase_start
                # Create the visualizations
                for viz_func in ['input_output', 'boxplot', 'multi_year_evolution']:
                    viz = self.get_visualization(viz_name=viz_func, inputs=x_input, labels=y_target, outputs=output)
                    wandb.log({'train/' + viz_func: wandb.Image(viz)}, commit=False)

                # Evaluate the validation dataset
                # if not self.debug:
                self.eval(data='val')
                current_val_loss = self.metrics['val']['loss'].compute()

                commit = step < n_iterations or (not self.use_early_stopping)
                self.log(step=step, phase_runtime=phase_runtime, commit=commit)
                self.reset_averaged_metrics()
                phase_start = time.time()

                # Check for early stopping
                if self.use_early_stopping:
                    if current_val_loss < self.best_val_loss:
                        self.best_val_loss = current_val_loss
                        self.best_model_path = self.save_model(model_identifier='best', sync=False)

        # Reload the best model if early stopping was used
        if self.use_early_stopping and self.best_model_path is not None:
            self.model = self.get_model(reinit=False, model_path=self.best_model_path)
            self.eval(data='best_val')
            self.log_best_model_metrics()

    def log_best_model_metrics(self):
        """Logs the best model's metrics after training."""
        loggingDict = {}
        # Add metrics
        split = 'best_val'
        for metric_name, metric in self.metrics[split].items():
            loggingDict[f"{split}/{metric_name}"] = metric.compute()

        # Log and push to Wandb
        for metric_type, val in loggingDict.items():
            wandb.run.summary[f"{metric_type}"] = val

        wandb.log(loggingDict, commit=True) # Now we commit

    def run(self):
        """Controls the execution of the script."""
        # We start training from scratch
        self.set_seed(seed=self.seed)  # Set the seed
        loaders = self.get_dataloaders()
        self.loader['train'], self.loader['val'], self.loader['fixed_val'] = loaders
        self.model = self.get_model(reinit=True, model_path=self.model_paths['initial'])  # Load the model

        self.define_optimizer_scheduler()  # This was moved before define_strategy to have the optimizer available

        self.train()  # Train the model
        self.eval(data='best_val')

        # Save the trained model and upload it to wandb
        self.save_model(model_identifier='trained', sync=True)