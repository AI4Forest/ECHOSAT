import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import ndimage


class SimplifiedSatelliteDataset(Dataset):
    """Simplified dataset class with hardcoded configuration."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        
        # Hardcoded configuration
        self.use_sentinel1 = True
        self.use_alos = True
        self.use_tandemx_dem = True
        self.use_tandemx_fnf = True
        self.gedi_filtering = True
        self.dem_source = "tandemx"
        self.urban_proportion_filter = "exclude"
        self.dem_slope_filter = 20
        self.urban_proportion_threshold = 10
        self.use_reduced_input_size = 96
        
        # Scaling dictionary
        self.scaling_dict = {
            (1, 2, 3, 4): (0, 2000),
            (6, 7, 8, 9): (0, 6000),
            (0,): (0, 1000),
            (5, 10, 11): (0, 4000),
        }
        
        # Load metadata and select fixed number of samples
        df = pd.read_csv(os.path.join(data_path, "metadata.csv"))     
        
        df = df.query('sample_id != 0').copy()
        
        # Build file paths
        df.loc[:, "files"] = df["tile"].apply(lambda x: os.path.join(data_path, "samples", x, x))
        df.loc[:, "files"] = df["files"] + "_" + df["sample_id"].astype(str) + ".npz"
        df = df.sample(frac=1).reset_index(drop=True)
        self.files = np.array(df["files"]).astype(np.bytes_)
        
        print(f"Loaded {len(self.files)} samples from {data_path}")
    
    def calculate_slope_degrees(self, elevation_data, invalid_value=-9999):
        """Calculate slope in degrees using min/max elevation in 3x3 grid."""
        valid_mask = elevation_data != invalid_value
        elev_copy = elevation_data.astype(np.float32)
        elev_copy[~valid_mask] = np.nan
        
        pixel_window = 7
        min_elev = ndimage.minimum_filter(elev_copy, size=pixel_window, mode='constant', cval=np.nan)
        max_elev = ndimage.maximum_filter(elev_copy, size=pixel_window, mode='constant', cval=np.nan)
        
        elev_diff = max_elev - min_elev
        distance = pixel_window * 10
        slope_radians = np.arctan(elev_diff / distance)
        slope_degrees = np.degrees(slope_radians)
        
        slope_degrees = np.nan_to_num(slope_degrees, nan=0.0)
        slope_degrees[~valid_mask] = 0
        
        return slope_degrees
    
    def apply_mask_to_gedi(self, gedi_data, mask, value=0):
        """Apply a 2D mask to GEDI data."""
        for year in range(gedi_data.shape[0]):
            for track in range(gedi_data.shape[1]):
                valid_mask = mask & (gedi_data[year, track] != 0)
                gedi_data[year, track][valid_mask] = value
    
    def apply_gedi_filtering(self, gedi_data, tandemx_dem_data, gedi_supplemental_data):
        """Apply GEDI filtering using slope and urban proportion filters."""
        filtered_gedi = gedi_data.copy()
        
        # Apply slope filtering
        if self.dem_source == "tandemx" and tandemx_dem_data is not None:
            slope_degrees = self.calculate_slope_degrees(tandemx_dem_data)
            if not np.all(np.isnan(slope_degrees)):
                steep_mask = slope_degrees > self.dem_slope_filter
                self.apply_mask_to_gedi(filtered_gedi, steep_mask, value=0)
        
        # Apply urban proportion filtering
        if gedi_supplemental_data is not None:
            urban_proportion = gedi_supplemental_data[4]  # Shape: (7, 6, 256, 256)
            max_urban_proportion = np.max(urban_proportion, axis=(0, 1))  # Shape: (256, 256)
            
            if not np.all(np.isnan(max_urban_proportion)):
                urban_mask = max_urban_proportion > self.urban_proportion_threshold
                if self.urban_proportion_filter == "exclude":
                    self.apply_mask_to_gedi(filtered_gedi, urban_mask, value=0)
                elif self.urban_proportion_filter == "zero":
                    self.apply_mask_to_gedi(filtered_gedi, urban_mask, value=2.5)
        
        return filtered_gedi
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        try:
            file = self.files[index]
            with np.load(file) as data:  # Use context manager to auto-close file
                return self._process_data(data)
        except Exception as e:
            print("FAILEDHERE", index, self.files[index])
            return self.__getitem__(random.randint(0, len(self.files)))
    
    def _process_data(self, data):
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
        
        return image_tensor, label_tensor


def create_dataloaders(data_path: str, batch_size: int = 2, num_workers: int = 8):
    """Create train and validation dataloaders."""
    
    # Create full dataset
    full_dataset = SimplifiedSatelliteDataset(data_path)
    
    # Split into train/val (90%/10%)
    val_size = max(min(int(0.1 * len(full_dataset)), 1000), 1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                            generator=torch.Generator().manual_seed(42))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, val_loader
