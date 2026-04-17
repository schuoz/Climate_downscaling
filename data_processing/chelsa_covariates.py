import xarray as xr
import numpy as np

class CovariateGenerator:
    """
    Generates high-resolution topographical and meteorological covariates 
    using CHELSA deterministic techniques, ported to native Python / xarray.
    These covariates are used as condition channels for the Consistency Model.
    """
    def __init__(self, dem_path: str):
        """
        :param dem_path: Path to the 100m DEM file (e.g. netcdf or tif)
        """
        # We load DEM natively using rioxarray or xarray
        if dem_path.endswith('.tif'):
            import rioxarray
            self.dem = rioxarray.open_rasterio(dem_path).squeeze('band')
        else:
            self.dem = xr.open_dataarray(dem_path)
            
    def compute_lapse_rate_temp(self, coarse_tas: xr.DataArray, coarse_dem: xr.DataArray, coarse_tlapse: xr.DataArray) -> xr.DataArray:
        """
        Calculate target 100m temperature assuming a linear lapse rate.
        This provides a strong physical prior for the ML model.
        """
        # Interpolate coarse grids to 100m resolution
        tas_100m = coarse_tas.interp_like(self.dem, method='linear')
        dem_low_100m = coarse_dem.interp_like(self.dem, method='linear')
        tlapse_100m = coarse_tlapse.interp_like(self.dem, method='linear')
        
        # Lapse rate based downscaling: T_high = T_low + (DEM_high - DEM_low) * tlapse
        # tlapse is usually in K/m
        tas_high = tas_100m + (self.dem - dem_low_100m) * tlapse_100m
        return tas_high

    def compute_wind_effect(self, u_wind: xr.DataArray, v_wind: xr.DataArray, wind_exposure: xr.DataArray) -> xr.DataArray:
        """
        Combines directional ERA5 wind with high-resolution wind exposition index.
        Note: The true CHELSA algorithm uses SAGA-GIS Multi-Level B-Splines 
        and Polar transformations. Here we provide a native xarray approximation.
        """
        # Interpolate wind components
        u_100m = u_wind.interp_like(self.dem, method='linear')
        v_100m = v_wind.interp_like(self.dem, method='linear')
        
        # Wind direction and speed
        wind_dir = np.arctan2(v_100m, u_100m)
        
        # Approximate wind effect leveraging a static exposure grid 
        # (exposure grid should ideally be precomputed using SAGA GIS 'Wind Exposition Index')
        wind_effect = np.abs(np.cos(wind_dir)) * wind_exposure  # Simplified proxy
        return wind_effect
        
    def get_all_covariates(self, env_data: xr.Dataset, static_exposure: xr.DataArray) -> xr.Dataset:
        """
        Generate all physics-guided input channels for the UNet.
        """
        tas_prior = self.compute_lapse_rate_temp(env_data['tas'], env_data['orog'], env_data['tlapse'])
        wind_prior = self.compute_wind_effect(env_data['u'], env_data['v'], static_exposure)
        
        covariates = xr.Dataset({
            'dem': self.dem,
            'tas_prior': tas_prior,
            'wind_prior': wind_prior
        })
        return covariates
