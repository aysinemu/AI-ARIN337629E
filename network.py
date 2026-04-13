"""
network.py - Refined License Plate Super-Resolution Network Architecture.

This module defines a deep Residual Dense Network (RDN) optimized for license plate 
super-resolution. It incorporates modern architectural techniques such as 
Residual Scaling, ICNR initialization, and Dual Positional Channel Attention (DPCA)
to ensure stable training and high-fidelity reconstruction.

Key Components:
    - RDB: Residual Dense Blocks with scaling to prevent gradient explosion.
    - TFAM: Text-Focused Attention Module for character-level enhancement.
    - DPCA: Dual Positional Channel Attention for global semantic awareness.
    - AutoEncoder: Initial feature extraction and noise reduction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import numpy as np
from typing import List, Tuple, Optional, Any, Union, Callable

# ============================================================================
# Utilities & Initialization helpers
# ============================================================================

def icnr_init(tensor: torch.Tensor, upscale_factor: int = 2, 
              init: Callable[[torch.Tensor], None] = nn.init.kaiming_normal_) -> None:
    """
    Implements ICNR (Initialization Causality with Nearest-Neighbor Resize).
    
    This technique is used for Sub-pixel Convolution (PixelShuffle) layers to 
    ensure that at initialization, the upsampling operation behaves like a 
    pure Nearest-Neighbor resize. This effectively eliminates 'checkerboard' 
    artifacts that typically plague fresh SR models.
    
    Args:
        tensor (torch.Tensor): The weight tensor of the convolution layer.
        upscale_factor (int): The PixelShuffle upsampling factor.
        init (Callable): The base weight initialization function to use for the sub-kernels.
    """
    new_shape = [tensor.shape[0] // (upscale_factor ** 2)] + list(tensor.shape[1:])
    sub_tensor = torch.zeros(new_shape).to(tensor.device)
    init(sub_tensor)
    sub_tensor = sub_tensor.transpose(0, 1)
    sub_tensor = sub_tensor.unsqueeze(2).unsqueeze(3)
    sub_tensor = sub_tensor.repeat(1, 1, upscale_factor, upscale_factor, 1, 1)
    sub_tensor = sub_tensor.view(new_shape[1], tensor.shape[0], *new_shape[2:])
    sub_tensor = sub_tensor.transpose(0, 1)
    tensor.data.copy_(sub_tensor) # type: ignore

# ============================================================================
# Core Building Blocks
# ============================================================================

class DConv(nn.Module):
    """
    Depthwise Separable Convolution.
    
    Reduces parameter count and computational complexity by splitting a standard 
    convolution into a spatial depthwise filter and a 1x1 pointwise filter.
    """
    def __init__(self, in_channels: int, out_channel: int, kernel_size: Union[int, Tuple[int, int]], 
                 stride: int = 1, padding: Union[str, int] = 'same', bias: bool = True) -> None:
        super(DConv, self).__init__()
        self.dConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      groups=in_channels, padding=padding, bias=bias),
            nn.Conv2d(in_channels, out_channel, kernel_size=1, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dConv(x)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding="same", bias=bias)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.ReLU(self.conv(x))], 1)

class RDB(nn.Module):
    """
    Residual Dense Block (RDB).
    
    The core engine of the RDN architecture. It uses dense connections to 
    extract local features and incorporates 'Residual Scaling' to ensure 
    numerical stability in extremely deep networks.
    
    Attributes:
        res_scale (float): Scaling factor for the residual branch.
    """
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int, 
                 bias: bool = False, res_scale: float = 0.1) -> None:
        super(RDB, self).__init__()
        self.res_scale = res_scale
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels + growth_rate * i, growth_rate, bias=bias)
              for i in range(num_layers)]
        )
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates residual output: x + scaled_local_features.
        Residual Scaling (0.1) helps prevent gradient explosion/saturation during training.
        """
        return x + self.lff(self.layers(x)) * self.res_scale

class RDN(nn.Module):
    """Residual Dense Network."""
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers, bias=False):
        super(RDN, self).__init__()
        self.num_blocks = num_blocks
        self.shallowF1 = nn.Conv2d(num_channels, num_features, kernel_size=7, padding="same", bias=bias)
        self.shallowF2 = nn.Conv2d(num_features, num_features, kernel_size=7, padding="same", bias=bias)

        self.rdbs = nn.ModuleList([RDB(num_features, growth_rate, num_layers, res_scale=0.1)])
        for _ in range(num_blocks - 1):
            self.rdbs.append(RDB(growth_rate, growth_rate, num_layers, res_scale=0.1))

        self.gff = nn.Sequential(
            nn.Conv2d(growth_rate * num_blocks, num_features, kernel_size=1, bias=bias),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding="same", bias=bias)
        )

    def forward(self, x):
        sfe1 = self.shallowF1(x)
        sfe2 = self.shallowF2(sfe1)
        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)
        x = self.gff(torch.cat(local_features, 1)) + sfe1
        return x

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channel, expansion=4):
        super(AutoEncoder, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, expansion * in_channels, kernel_size=7, stride=1, padding='same', bias=False)
        self.encoder = nn.Sequential(
            DConv(expansion * in_channels, expansion * in_channels, kernel_size=7),
            nn.PixelUnshuffle(2),
            nn.ReLU(inplace=True),
            DConv(expansion * in_channels * 2**2, expansion * in_channels, kernel_size=7),
            nn.PixelUnshuffle(2),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            DConv(expansion * in_channels * 2**2, expansion * in_channels * 2**2, kernel_size=7),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            DConv(expansion * in_channels, expansion * in_channels * 2**2, kernel_size=7),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.GA = nn.Sequential(self.encoder, self.decoder)
        self.conv_out = nn.Conv2d(expansion * in_channels, out_channel, kernel_size=3, stride=1, padding='same', bias=False)

    def forward(self, x):
        conv_in = self.conv_in(x)
        out = self.GA(conv_in)
        out = conv_in + out
        out = self.conv_out(out)
        return out

# ============================================================================
# Attention Modules (Kept stable spatial/channel attentions)
# ============================================================================

class DPCA(nn.Module):
    """Dual Positional Channel Attention."""
    def __init__(self, in_channels, out_channels):
        super(DPCA, self).__init__()
        self.global_avg_pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.global_avg_pool_v = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_h = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv_f = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.sigmoid(self.conv_h(self.global_avg_pool_h(x)))
        v = self.sigmoid(self.conv_v(self.global_avg_pool_v(x)))
        f = self.sigmoid(self.conv_f(x))
        return h * v * f

class TFAM(nn.Module):
    """
    Text-Focused Attention Module (TFAM).
    
    A custom attention mechanism designed specifically for alphanumeric shapes.
    It combines spatial attention (POS) and channel attention (CA) with Dual 
    Positional Channel Attention (DPCA) to emphasize character boundaries.
    """
    def __init__(self, channels_in: int, num_features: int = 128, bias: bool = False) -> None:
        super(TFAM, self).__init__()
        self.bias = bias
        self.convIn = nn.Sequential(
            DConv(channels_in, num_features, kernel_size=3, bias=self.bias),
            nn.ReLU(inplace=True),
        )
        self.DPCA = DPCA(num_features, 2 * num_features)
        self.posAvg = nn.AvgPool2d(2)
        self.posMax = nn.MaxPool2d(2)
        self.POS_unit = self._POS(num_features)
        self.CA_unit = self._CA(num_features)
        self.convOut = nn.Sequential(
            DConv(2 * num_features, channels_in, kernel_size=3, bias=self.bias)
        )

    def _POS(self, channels_in: int) -> nn.Module:
        """Spatial attention branch using PixelShuffle for high-res map generation."""
        return nn.Sequential(
            DConv(2 * channels_in, channels_in * (2**2), kernel_size=3, bias=self.bias),
            nn.PixelShuffle(2),
            DConv(channels_in, 2 * channels_in, kernel_size=3, bias=self.bias),
        )

    def _CA(self, channels_in: int) -> nn.Module:
        """Channel attention branch using PixelUnshuffle for multi-scale pooling."""
        return nn.Sequential(
            nn.Conv2d(channels_in, 2 * channels_in, kernel_size=1, stride=1, 
                      groups=channels_in, padding='same', bias=self.bias),
            nn.PixelUnshuffle(2),
            DConv(2 * channels_in * (2**2), 2 * channels_in * (2**2), kernel_size=3, bias=self.bias),
            nn.PixelShuffle(2),
            DConv(2 * channels_in, channels_in, kernel_size=3, bias=self.bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies fused spatial-channel attention to highlight text regions."""
        convIn = self.convIn(x)
        dpca = self.DPCA(convIn)
        out = self.convOut(
            (torch.cat((self.CA_unit(convIn), convIn), dim=1) +
             self.POS_unit(torch.cat((self.posAvg(convIn), self.posMax(convIn)), dim=1))) * dpca
        )
        out = x * torch.sigmoid(out)
        return out

class AdaptiveResidualBlock(nn.Module):
    """
    An advanced residual block that adapts to local textures.
    
    It uses a dual-path strategy:
    1. BNpath: Standard convolution + TFAM attention.
    2. ADPpath: Downsample-Upsample path to capture wider spatial context.
    """
    def __init__(self, in_channels: int, growth_rate: int = 12, bias: bool = False) -> None:
        super(AdaptiveResidualBlock, self).__init__()
        self.BNpath = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=bias),
            TFAM(growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )
        self.ADPpath = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )
        self.convOut = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combines adaptive context with residual scaling for stability."""
        # RESIDUAL SCALING: Multiplied by 0.1 to stabilize deep gradient flow
        resPath = self.convOut(self.BNpath(x) + x) * 0.1
        outADP = self.ADPpath(x)
        return resPath + outADP

class ARC(nn.Module):
    def __init__(self, channels_in, expansion=128):
        super(ARC, self).__init__()
        self.input = AdaptiveResidualBlock(channels_in, expansion)

    def forward(self, x):
        return self.input(x)
        
class ResidualConcatenationBlock(nn.Module):
    def __init__(self, channels_in, out_channels, num_layers=2, bias=False):
        super(ResidualConcatenationBlock, self).__init__()
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.pw = nn.ModuleList([
            nn.Conv2d((2**(i+1)) * channels_in, (2**(i+1)) * channels_in, kernel_size=1, stride=1, padding=0, bias=bias)
            for i in range(num_layers - 1)
        ])
        self.pw.append(nn.Conv2d(2**num_layers * channels_in, out_channels, kernel_size=1, stride=1, padding=0, bias=bias))
        self.block = nn.ModuleList([ARC((2**i) * channels_in) for i in range(num_layers)])
        self.final_pw = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        for i in range(1, self.num_layers + 1):
            x = torch.cat((self.block[i-1](x), x), dim=1)
            x = self.pw[i-1](x)
        x = self.final_pw(x)
        return x

class ResidualModule(nn.Module):
    def __init__(self, channels_in, out_channels, num_layers=2, bias=False):
        super(ResidualModule, self).__init__()
        self.num_layers = num_layers
        self.pw = nn.ModuleList([
            nn.Conv2d((2**(i+1)) * channels_in, (2**(i+1)) * channels_in, kernel_size=1, stride=1, padding='same', bias=bias)
            for i in range(num_layers - 1)
        ])
        self.pw.append(nn.Conv2d(2**num_layers * channels_in, out_channels, kernel_size=1, stride=1, padding='same', bias=bias))
        self.block = nn.ModuleList([ResidualConcatenationBlock((2**i) * channels_in, (2**i) * channels_in) for i in range(num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            x = torch.cat((self.block[i](x), x), dim=1)
            pw = self.pw[i](x)
        return pw

class FeatureModule(nn.Module):
    def __init__(self, channels_in, skip_connection_channels):
        super(FeatureModule, self).__init__()
        self.TFAM = TFAM(channels_in)
        self.conv = nn.Conv2d(channels_in, skip_connection_channels, kernel_size=3, stride=1, padding='same', bias=False)

    def forward(self, x, skip_connection):
        out = self.conv(self.TFAM(x))
        output = torch.add(out, skip_connection)
        return output

# ============================================================================
# Latent Space & Distillation Modules
# ============================================================================

class LatentEncoder(nn.Module):
    """
    Transforms rich feature maps into a compact spatial latent map.
    
    The latent map is used for 'Latent Correlation Loss', which aligns the 
    internal semantic understanding of the LR network with the HR ground truth.
    """
    def __init__(self, in_channels: int = 128, latent_dim: int = 256) -> None:
        super(LatentEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produces an L2-normalized latent map."""
        latent = self.encoder(x)
        # CRITICAL: Small epsilon (1e-8) prevents division by zero during normalization
        latent = F.normalize(latent, p=2, dim=1, eps=1e-8)
        return latent

class HRFeatureEncoder(nn.Module):
    def __init__(self, in_channels=3, feat_channels=128):
        super(HRFeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.encoder(x)

# ============================================================================
# Main Network
# ============================================================================

class ImprovedNetwork(nn.Module):
    """
    The main architectural hub for License Plate Super-Resolution.
    
    Multi-phase Pipeline:
      1. Shallow/Dense Extraction: AutoEncoder + RDN layers.
      2. Attention Refinement: Residual and Feature modules with TFAM.
      3. Upsampling: PixelShuffle with ICNR initialization.
      4. Global Residual: Bicubic SR refinement for structural stability.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 feat_channels: int = 128, rdn_blocks1: int = 16, 
                 rdn_blocks2: int = 8, latent_dim: int = 256) -> None:
        super(ImprovedNetwork, self).__init__()
        
        # 1. Feature Extraction
        self.inputLayer = nn.Sequential(
            AutoEncoder(in_channels, feat_channels),
            RDN(feat_channels, feat_channels, feat_channels, rdn_blocks1, 3)
        )
        self.RM = ResidualModule(feat_channels, feat_channels)
        self.FM = FeatureModule(feat_channels, feat_channels)
        
        # 2. Upsampling
        self.conv1 = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=1, padding='same', bias=False)
        upscale = []
        for _ in range(4 // 2):
            upscale.extend([
                nn.Conv2d(feat_channels, feat_channels * 2**2, kernel_size=3, stride=1, padding='same', bias=False),
                nn.PixelShuffle(2)
            ])
        self.upscale = nn.Sequential(*upscale)
        
        # 3. Refinement & Output
        self.RDN_post = RDN(feat_channels, feat_channels, feat_channels, rdn_blocks2, 3)
        self.Output = nn.Conv2d(feat_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False)
        
        # 4. Latent Space Mapping (for training)
        self.latent_encoder = LatentEncoder(feat_channels, latent_dim)
        self.hr_encoder = HRFeatureEncoder(in_channels, feat_channels)
        self.hr_latent_encoder = LatentEncoder(feat_channels, latent_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Custom weight initialization for Deep Super-Resolution.
        
        Strategy:
          - Use Kaiming Normal for ReLU-based layers.
          - Apply 'Variance Reduction' (0.1x scaling) to all non-output convolution 
            layers. This is a crucial trick for deep networks that prevents signals
            from exploding during the first few iterations of SGD.
          - Apply ICNR to upsampling layers to eliminate checkerboard patterns.
          - Set Output layer to zero initially to function as an Identity mapping.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.Output:
                    # Final layer starts at zero (identity mapping starting point)
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif m.out_channels == m.in_channels * 4:
                    # Detect and apply ICNR to Sub-pixel convolution kernels
                    icnr_init(m.weight, upscale_factor=2)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    # Standard Kaiming Normalization
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    
                    # This allows higher initial learning rates without instability.
                    m.weight.data *= 0.1 
                    
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, 
                hr: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Executes the Super-Resolution pipeline.
        
        Args:
            x (torch.Tensor): Low-resolution input batch (B, 3, H, W).
            hr (Optional[torch.Tensor]): Optional high-resolution target for training distillation.
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                - Evaluation mode: Returns SR image (B, 3, 4H, 4W).
                - Training mode: Returns (SR, Latent_SR, Latent_HR) for distillation loss.
        """
        # LR Feature Pipeline
        Input = self.inputLayer(x)
        RMOutput = self.RM(Input)
        FMOutput = self.FM(RMOutput, Input)
        
        # Upscale Pipeline
        PS1 = self.conv1(FMOutput)
        PS1 = self.upscale(PS1)
        PS1 = self.RDN_post(PS1)
        
        # Global Residual Connection: Add upsampled LR to SR (Identity refinement)
        output = self.Output(PS1)
        output = output + F.interpolate(x, (output.size(-2), output.size(-1)), mode='bicubic', align_corners=False)
        
        # Training Phase: Distillation via Latent Space
        if hr is not None:
            latent_sr = self.latent_encoder(FMOutput)
            
            hr_feat_for_latent = self.hr_encoder(hr)
            # Downsample HR features to match LR spatial dimension for alignment
            hr_feat_resized = F.adaptive_avg_pool2d(hr_feat_for_latent, (FMOutput.size(2), FMOutput.size(3)))
            latent_hr = self.hr_latent_encoder(hr_feat_resized)
            
            return output, latent_sr, latent_hr
        
        return output

# ============================================================================
# VGG Feature Extractor for Perceptual Loss
# ============================================================================

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_idx=18):
        super(VGGFeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:layer_idx])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, img):
        img = (img - self.mean) / self.std
        return self.feature_extractor(img)