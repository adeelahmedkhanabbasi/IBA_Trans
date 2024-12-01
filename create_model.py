from .IBA import ViT3D
from timm import create_model
import torchvision.models as models
import torch
import torch.nn as nn
from torch.nn import functional as F

def create_model(model, img_size, n_classes, d=80):
    if model == 'vit3D':
        patch_size = 16
        heads = 12
        sd = 0.1
        DMSSCE = 9
        model = ViT3D(img_size=img_size, patch_size = patch_size,DMSSCE=DMSSCE, num_classes=n_classes, dim=192,
                    mlp_dim_ratio=2, channels=4, depth=12, heads=heads, dim_head=192//heads,
                    stochastic_depth=sd, is_FGRT=True, is_BFA=True)
    return model

