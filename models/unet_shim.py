import torch
import torch.nn as nn
import torch.nn.functional as F

class CovariateModulator(nn.Module):
    """
    A shim layer that precedes the standard Consistency/SDE Unet.
    It takes the high-dimensional input (ERA5 + HighRes Covariates)
    and projects it down/up to the channel dimension expected by the base model.
    This allows us to use the original OpenAI/Song UNet without rewriting its core.
    """
    def __init__(self, in_channels: int, base_unet_in_channels: int):
        super().__init__()
        # Project our N covariates + C ERA5 channels into the embedding space the UNet expects
        self.projection = nn.Conv2d(in_channels, base_unet_in_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        """
        x shape: (B, in_channels, H, W)
        Returns: (B, base_unet_in_channels, H, W)
        """
        return self.projection(x)

class PhysicsGuidedUNet(nn.Module):
    def __init__(self, base_unet: nn.Module, total_in_channels: int, unet_expected_channels: int):
        super().__init__()
        # The covariate modulator adjusts the input dimensionality
        self.modulator = CovariateModulator(total_in_channels, unet_expected_channels)
        self.base_unet = base_unet
        
    def forward(self, x, time_steps, *args, **kwargs):
        """
        x is the noisy target concatenated with our physics priors.
        """
        # We split x: target is the first channel, conditions are the rest
        # Actually usually Consistency models take target and condition separately,
        # or concatenated. Assume concatenated here.
        x_modulated = self.modulator(x)
        # Pass to the base UNet (e.g. NCSNpp or ConsistencyModel)
        out = self.base_unet(x_modulated, time_steps, *args, **kwargs)
        return out
