import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .FGRT3D import FGRT3D
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# classes

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, dropout=0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads=12, dim_head=64, dropout=0., is_BFA=False, proj_drop=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias=False)
        init_weights(self.to_qkv)
        self.is_BFA = is_BFA
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if is_BFA:
            self.scale = nn.Parameter(self.scale * torch.ones(heads))
            self.mask = torch.eye(self.num_patches + 1, self.num_patches + 1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        if self.is_BFA is False:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k),
                             scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            se = dots.mean(dim=1, keepdim=True)

            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = float('-inf')

            B, N, C = x.shape
            top_k = int(N * 0.75)
            mask1 = torch.zeros(B, self.heads, N, N, device=x.device, requires_grad=False)
            index = torch.topk(dots, k=top_k, dim=-1, largest=True)[1]
            mask1.scatter_(-1, index, 1.)

            dots = torch.where(mask1 > 0, dots, torch.full_like(dots, float('-inf')))

            dots += se

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)



class ChannelSELayer3D(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class SpatialSELayer3D(nn.Module):
    def __init__(self, channel):
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(channel, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply convolution to get a single channel output
        x = self.conv(x)
        # Apply sigmoid to get weights in the range [0, 1]
        x = self.sigmoid(x)
        return x


class DMSSCE3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., kernel_size=1,
                 with_bn=True, dim_head=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.scale = dim_head ** -0.5

        # Pointwise
        self.conv1 = nn.Conv3d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)

        # Depthwise dilated
        self.conv2 = nn.Conv3d(
            hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, dilation=2, groups=hidden_features)

        # Depthwise dilated convolutions with variable dilation rates
        self.dilated_convs = nn.ModuleList([
            nn.Conv3d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                      groups=hidden_features)
            for dilation in [1, 2, 4]  # Example dilation rates: 1, 2, 4
        ])

        # Channel SE Block
        self.cSE = ChannelSELayer3D(hidden_features)

        # Pointwise
        self.conv3 = nn.Conv3d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()

        self.bn = nn.ModuleList([nn.BatchNorm3d(hidden_features) for _ in range(len(self.dilated_convs))])
        self.bn1 = nn.BatchNorm3d(hidden_features)
        self.bn2 = nn.BatchNorm3d(hidden_features)
        self.bn3 = nn.BatchNorm3d(out_features)

        # Spatial SE Block
        self.sSE = SpatialSELayer3D(hidden_features)

        # The reduction ratio is always set to 4
        self.squeeze = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.compress = nn.Linear(in_features, in_features // 4)
        self.excitation = nn.Linear(in_features // 4, in_features)

    def forward(self, x):
        B, N, C = x.size()
        D = 5
        H = 8
        W = 8
        cls_token, tokens = torch.split(x, [1, N - 1], dim=1)
        x = tokens.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        shortcut = x

        dilated_features = []
        for conv, bn in zip(self.dilated_convs, self.bn):
            dilated_features.append(self.act(bn(conv(x))))
        x = sum(dilated_features) / len(dilated_features)

        x = shortcut + x

        # Apply spatial SE block
        spatial_attention = self.sSE(x)
        x = x * spatial_attention

        # Channel SE Block
        x = self.cSE(x)

        x = self.conv3(x)
        x = self.bn3(x)

        tokens = x.flatten(2).permute(0, 2, 1)
        out = torch.cat((cls_token, tokens), dim=1)

        return out

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, DMSSCE, dropout=0., stochastic_depth=0.,
                 is_BFA=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale = {}
        self.initials = DMSSCE
        self.depth = depth
        self.is_BFA = is_BFA


        for i in range(self.initials):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim,DMSSCE3D(in_features=dim, hidden_features=dim, dim_head=dim_head, drop=dropout)),
            ]))

        for i in range(self.depth - self.initials):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim,
                        attention(dim=dim, num_patches=num_patches, heads=heads, dim_head=dim_head, dropout=dropout,
                            is_BFA=is_BFA)),
                PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout=dropout))
            ]))

        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()

    def forward(self, x):
        skip_connection = x.clone()
        for i, layer in enumerate(self.layers):
            attn = layer[0]
            x = self.drop_path(attn(x)) + x
            if len(layer) > 1:  # Check if FeedForward exists in the layer
                ff = layer[1]
                x = self.drop_path(ff(x)) + x
            self.scale[str(i)] = attn.fn.scale

            if i == self.initials:  # Add skip connection after initial block
               x = x  + skip_connection

        return x


class ViT3D(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, DMSSCE, channels=4,
                 dim_head=16, dropout=0., emb_dropout=0., stochastic_depth=0., is_BFA=False, is_FGRT=False):
        super().__init__()
        image_depth, image_height, image_width = 80, img_size, img_size
        patch_depth, patch_height, patch_width = patch_size, patch_size, patch_size
        self.num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth)
        self.patch_dim = channels * patch_height * patch_width * patch_depth
        self.dim = dim
        self.num_classes = num_classes

        if not is_FGRT:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)', p1=patch_depth, p2=patch_height,
                          p3=patch_width),
                nn.Linear(self.patch_dim, self.dim)
            )

        else:
            self.to_patch_embedding = FGRT3D(channels, self.dim, patch_size, is_pe=True)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(self.dim, self.num_patches, depth, heads, dim_head, mlp_dim_ratio, DMSSCE,
                                        dropout,
                                        stochastic_depth, is_BFA=is_BFA)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )

        self.apply(init_weights)

    def forward(self, img):

        x = self.to_patch_embedding(img)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)


        return self.mlp_head(x[:, 0])

