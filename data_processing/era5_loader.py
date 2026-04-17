import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from typing import Optional

from .chelsa_covariates import CovariateGenerator

class DownscalingDataset(Dataset):
    """
    Combined dataloader that merges coarse ERA5 temporal data with 
    high-resolution CHELSA physical covariates (100m) for Physics-Guided ML.
    """
    def __init__(self, era5_path: str, dem_path: str, exposure_path: str, variables: list):
        self.era5_ds = xr.open_dataset(era5_path)
        self.covariate_gen = CovariateGenerator(dem_path)
        self.static_exposure = xr.open_dataarray(exposure_path)
        self.variables = variables
        
        # Dimensions
        self.times = self.era5_ds.time.values
        self.num_samples = len(self.times)
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Get ERA5 features for this timestep
        era5_slice = self.era5_ds.isel(time=idx)
        
        # 2. Compute the CHELSA physical high-res covariates (the 'priors')
        covariates = self.covariate_gen.get_all_covariates(era5_slice, self.static_exposure)
        
        # 3. Stack channels: [ERA5_tas, ERA5_pr, ..., DEM, TAS_Prior, WIND_Prior]
        # In a real setup, coarse ERA5 needs to be upsampled (e.g. nearest or bilinear) to match the dimensions of covariates
        era5_upsampled = era5_slice.interp_like(covariates.dem, method='nearest')
        
        channels = []
        # Add basic coarse features
        for var in self.variables:
            channels.append(era5_upsampled[var].values)
            
        # Add physics-guided covariates
        channels.append(covariates['dem'].values)
        channels.append(covariates['tas_prior'].values)
        channels.append(covariates['wind_prior'].values)
        
        # Stack to shape (C, H, W)
        stacked_tensor = np.stack(channels, axis=0)
        
        # Handle Nans
        stacked_tensor = np.nan_to_num(stacked_tensor, nan=0.0)
        
        # Convert to Torch
        stacked_tensor = torch.from_numpy(stacked_tensor).float()
        
        # If we had high-res targets (say, for training), we return (x, y)
        # Here we assume inference / feature building for the ML model
        # Target dummy
        y = torch.zeros((1, stacked_tensor.shape[1], stacked_tensor.shape[2]))
        
        return stacked_tensor, y

def create_dataloader(era5_path, dem_path, exposure_path, variables, batch_size=4):
    ds = DownscalingDataset(era5_path, dem_path, exposure_path, variables)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
