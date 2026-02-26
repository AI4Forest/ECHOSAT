from torch.utils.data import Dataset
import os
import torch
import numpy as np
import random
import pandas as pd
import sys


class SatelliteImageDataset4D(Dataset):
    """
    Dataset class for preprocessed satellite imagery that supports a 4D input for a U-Net.
    Each sample groups multiple years from the same region (identified by tile and region_id)
    so that the resulting image tensor has shape: (channels, years, months, H, W)
    """
    def __init__(self, data_path: str, fixed_val: bool = False, use_reduced_input_size: bool = False, use_overlapping_patches: bool = False, overlap_size: int = 40):
        self.data_path = data_path
        
        # Hardcoded configuration
        self.use_sentinel1 = True
        self.use_alos = True
        self.use_tandemx_dem = True
        self.use_tandemx_fnf = True
        self.gedi_filtering = False
        # self.dem_source = "tandemx"
        # self.urban_proportion_filter = "exclude"
        # self.dem_slope_filter = 20
        # self.urban_proportion_threshold = 10

        # Read the metadata CSV
        df = pd.read_csv(os.path.join(data_path, "metadata.csv"))
        self.fixed_val = fixed_val
        if 'fixed_val' in df.columns:
            if fixed_val:
                df = df.query("fixed_val == 1")
            else:
                df = df.query("fixed_val == 0")
        else:
            if fixed_val:
                df = df.sample(min(10, len(df)))     

        # Build file paths (assuming the same naming convention)
        df["files"] = df["tile"].apply(lambda x: os.path.join(data_path, "samples", x, x))
        df["files"] = df["files"] + "_" + df["sample_id"].astype(str) + ".npz"
        self.files = np.array(df["files"]).astype(np.bytes_)
        self.scaling_dict = self.get_scaling_dict()
        self.use_reduced_input_size = use_reduced_input_size
        self.use_overlapping_patches = use_overlapping_patches
        self.overlap_size = overlap_size
        
        assert not (self.use_reduced_input_size == 256 and self.use_overlapping_patches), "Must use reduced input size if using overlapping patches"
        
    def get_scaling_dict(self):
        base_dict = {
            (1, 2, 3, 4): (0, 2000),
            (6, 7, 8, 9): (0, 6000),
            (0,): (0, 1000),
            (5, 10, 11): (0, 4000),
        }
        sys.stdout.write(f"Scaling dict: {base_dict}.\n")
        return base_dict

    def __len__(self):
        # The length is now the number of tile-region groups
        return len(self.files)

    def __getitem__(self, index):
        try:
            file = self.files[index]
            with np.load(file) as data:  # Use context manager to auto-close file
                return self._process_data(data, file)
        except Exception as e:
            print("FAILEDHERE", index, self.files[index])
            return self.__getitem__(random.randint(0, len(self.files)))
    
    def _process_data(self, data, file):
        # Load data
        label_data = data['gedi'].astype(np.float32)
        sentinel2_data = data['sentinel2'].astype(np.float32)
        
        if "_v3_" in self.data_path:
            sentinel2_data = np.roll(sentinel2_data, 1, axis=0)
        
        # Load optional data
        if self.use_sentinel1:
            sentinel1_data = (data['sentinel1'].astype(np.float32) - 22500) / 1000
        if self.use_alos:
            alos_data = (10 * np.log10(data['alos'].astype(np.float32)[:, :2] ** 2)) - 83.0
        if self.use_tandemx_dem or self.gedi_filtering:
            tandemx_dem_data = data['tandemx_dem'].astype(np.float32)
        if self.use_tandemx_fnf:
            tandemx_fnf_data = data['tandemx_fnf'].astype(np.float32)
        
        gedi_supplemental_data = data['gedi_supplemental'].astype(np.float32)
        
        # Apply GEDI filtering
        if self.gedi_filtering:
            if all(x is not None for x in [tandemx_dem_data, gedi_supplemental_data]):
                label_data = self.apply_gedi_filtering(
                    label_data, tandemx_dem_data, gedi_supplemental_data
                )
        
        # Process and concatenate data
        to_concat = []
        years, months, s2_channels, height, width = sentinel2_data.shape
        
        # Scale Sentinel-2 data
        for channels, (min_val, max_val) in self.scaling_dict.items():
            for channel in channels:
                sentinel2_data[:, :, channel] = np.clip(sentinel2_data[:, :, channel], min_val, max_val)
                sentinel2_data[:, :, channel] = (sentinel2_data[:, :, channel] - min_val) / (max_val - min_val)
        to_concat.append(sentinel2_data)
        
        # Add optional data
        if self.use_sentinel1:
            _, quarters, s1_channels, _, _ = sentinel1_data.shape
            expanded_sentinel1 = np.zeros((years, months, s1_channels, height, width), dtype=np.float32)
            for month_idx in range(months):
                quarter_idx = int(np.floor(month_idx * quarters / months))
                expanded_sentinel1[:, month_idx, :, :, :] = sentinel1_data[:, quarter_idx, :, :, :]
            expanded_sentinel1 = (expanded_sentinel1 + 50) / 51
            expanded_sentinel1 = np.clip(expanded_sentinel1, 0, 1)
            to_concat.append(expanded_sentinel1)
        
        if self.use_alos:
            expanded_alos = np.tile(alos_data[:, np.newaxis], (1, months, 1, 1, 1))
            expanded_alos = (expanded_alos + 50) / 51
            expanded_alos = np.clip(expanded_alos, 0, 1)
            to_concat.append(expanded_alos)
        
        if self.use_tandemx_dem:
            expanded_tandemx_dem = np.tile(tandemx_dem_data[np.newaxis, np.newaxis, np.newaxis], (years, months, 1, 1, 1))
            expanded_tandemx_dem = expanded_tandemx_dem / 7000
            expanded_tandemx_dem = np.clip(expanded_tandemx_dem, 0, 1)
            to_concat.append(expanded_tandemx_dem)
        
        if self.use_tandemx_fnf:
            expanded_tandemx_fnf = np.tile(tandemx_fnf_data[np.newaxis, np.newaxis, np.newaxis], (years, months, 1, 1, 1))
            expanded_tandemx_fnf = expanded_tandemx_fnf / 3
            expanded_tandemx_fnf = np.clip(expanded_tandemx_fnf, 0, 1)
            to_concat.append(expanded_tandemx_fnf)
        
        # Concatenate all data
        combined_data = np.concatenate(to_concat, axis=2)
        
        # Scale to [-1, 1]
        combined_data = (combined_data * 2) - 1
        
        # Apply center cropping for reduced input size
        if self.use_reduced_input_size > 0:
            crop_size = self.use_reduced_input_size
            start_h = (height - crop_size) // 2
            end_h = start_h + crop_size
            start_w = (width - crop_size) // 2
            end_w = start_w + crop_size
            
            combined_data = combined_data[:, :, :, start_h:end_h, start_w:end_w]
            label_data = label_data[:, :, start_h:end_h, start_w:end_w]
        
        # Convert to tensors and permute to (channels, years, months, H, W)
        image_tensor = torch.tensor(combined_data)
        image_tensor = image_tensor.permute(2, 0, 1, 3, 4)  # (channels, years, months, H, W)
        label_tensor = torch.tensor(label_data)
        
        # Explicit cleanup to help with memory management
        del combined_data
        del to_concat
        if self.fixed_val:
            return image_tensor, label_tensor, os.path.basename(file)
        return image_tensor, label_tensor

        

    