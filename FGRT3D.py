import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        y = nn.functional.adaptive_avg_pool3d(x, 1)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y
# Define the Edge and Shape Feature Extraction Block for 3D
class EdgeAndShapeFeatureBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeAndShapeFeatureBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        fused_features = x2
        return fused_features

# Define the PatchShifting module for 3D
class PatchShifting3D(nn.Module):
    def __init__(self, patch_size, in_dim, dim):
        super().__init__()
        
        self.shift = int(patch_size * (1 / 2))
        self.edge_shape_block = EdgeAndShapeFeatureBlock3D(in_dim, in_dim)  # Adjusted to match input channels
        self.se_block = SEBlock(in_dim)  # SE block for combined features
        self.convemd = nn.Conv3d(in_dim*3, in_dim*3, kernel_size=16,stride=8,padding=7)
        self.convemd2 = nn.Conv3d(in_dim*3, in_dim*3, kernel_size=2, stride=2,
                                 padding=0)

    def forward(self, x):
        shortcut = x
        x = self.edge_shape_block(x)
        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift, 0, 0))

        x_ul = x_pad[:, :, :, :-self.shift * 2, :-self.shift * 2]
        x_ur = x_pad[:, :, :, self.shift * 2:, :-self.shift * 2]
        x_ll = x_pad[:, :, :, :-self.shift * 2, :-self.shift * 2]
        x_lr = x_pad[:, :, :, self.shift * 2:, :-self.shift * 2]
        x_cat = torch.cat([shortcut, x_ur, x_ll], dim=1)
        return x_cat

# Define the FGRT3D module for 3D
class FGRT3D(nn.Module):
    def __init__(self, in_dim, dim, merging_size=2, exist_class_t=False, is_pe=False):
        super().__init__()
        
        self.exist_class_t = exist_class_t
        self.patch_shifting = PatchShifting3D(merging_size, in_dim, dim)
        patch_dim = (in_dim * 3) * ((merging_size) * merging_size * merging_size )
        if exist_class_t:
            self.class_linear = nn.Linear(in_dim, dim)

        self.is_pe = is_pe
        """
        self.merging = nn.Sequential(
            Rearrange('b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)', p1=merging_size, p2=merging_size, p3=merging_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )
        """
        self.merging = nn.Sequential(
            #Rearrange('b d h w c -> b (d h w) c'),
            #nn.Linear(in_dim*3, dim)
            Rearrange('b (d p1) (h p2) (w p3) c-> b (d h w) (p1 p2 p3 c)', p1=(merging_size) , p2=merging_size , p3=merging_size),
            nn.Linear(patch_dim , dim)
        )

    def forward(self, x):
        if self.exist_class_t:
            visual_tokens, class_token = x[:, 1:], x[:, (0,)]
            reshaped = rearrange(visual_tokens, 'b (d h w) d -> b d d h w', d=int(math.sqrt(x.size(1))))
            out_visual = self.patch_shifting(reshaped)
            out_visual = self.merging(out_visual)
            out_class = self.class_linear(class_token)
            out = torch.cat([out_class, out_visual], dim=1)
        else:
            out = x if self.is_pe else rearrange(x, 'b (d h w) d -> b d d h w', d=int(round(x.size(1) ** (1/3))))
            out = self.patch_shifting(out)
            out= out.permute(0, 2, 3, 4, 1)
            out = self.merging(out)

        return out