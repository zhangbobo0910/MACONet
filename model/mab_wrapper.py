import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 基础层 ---------- #
class LayerNorm(nn.Module):
    """Channel‑first LayerNorm  (B, C, H, W)."""
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_first'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps
        assert data_format in ('channels_last', 'channels_first')
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape,
                                self.weight, self.bias, self.eps)
        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GSAU(nn.Module):
    """Gated Spatial Attention Unit."""
    def __init__(self, n_feats):
        super().__init__()
        inter_feats = n_feats * 2
        self.conv1  = nn.Conv2d(n_feats, inter_feats, 1)
        self.dwconv = nn.Conv2d(n_feats, n_feats, 7, padding=3, groups=n_feats)
        self.conv2  = nn.Conv2d(n_feats, n_feats, 1)
        self.norm   = LayerNorm(n_feats, data_format='channels_first')
        self.scale  = nn.Parameter(torch.zeros((1, n_feats, 1, 1)))

    def forward(self, x):
        shortcut = x
        x = self.conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)        # Ghost branch
        x = x * self.dwconv(a)
        x = self.conv2(x)
        return x * self.scale + shortcut


class MLKA(nn.Module):
    """Multi‑scale Large‑Kernel Attention."""
    def __init__(self, n_feats):
        super().__init__()
        assert n_feats % 3 == 0, f'n_feats must be ÷3, got {n_feats}'
        s = n_feats // 3
        self.norm  = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)))

        self.lka3 = nn.Sequential(
            nn.Conv2d(s, s, 3, padding=1, groups=s),
            nn.Conv2d(s, s, 5, padding=4, dilation=2, groups=s),
            nn.Conv2d(s, s, 1))
        self.lka5 = nn.Sequential(
            nn.Conv2d(s, s, 5, padding=2, groups=s),
            nn.Conv2d(s, s, 7, padding=9, dilation=3, groups=s),
            nn.Conv2d(s, s, 1))
        self.lka7 = nn.Sequential(
            nn.Conv2d(s, s, 7, padding=3, groups=s),
            nn.Conv2d(s, s, 9, padding=16, dilation=4, groups=s),
            nn.Conv2d(s, s, 1))

        self.x3 = nn.Conv2d(s, s, 3, padding=1, groups=s)
        self.x5 = nn.Conv2d(s, s, 5, padding=2, groups=s)
        self.x7 = nn.Conv2d(s, s, 7, padding=3, groups=s)

        self.proj_first = nn.Conv2d(n_feats, n_feats * 2, 1)
        self.proj_last  = nn.Conv2d(n_feats, n_feats, 1)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)        # attention branch / feature branch
        a1, a2, a3 = torch.chunk(a, 3, dim=1)
        attn = torch.cat([
            self.lka3(a1) * self.x3(a1),
            self.lka5(a2) * self.x5(a2),
            self.lka7(a3) * self.x7(a3)
        ], dim=1)
        x = self.proj_last(x * attn)
        return x * self.scale + shortcut


class MAB(nn.Module):
    """Multi‑scale Attention Block  (MLKA ➜ GSAU)."""
    def __init__(self, n_feats):
        super().__init__()
        self.lka  = MLKA(n_feats)
        self.gsau = GSAU(n_feats)

    def forward(self, x):
        x = self.lka(x)
        x = self.gsau(x)
        return x



class MABWrapper(nn.Module):

    def __init__(self, channels):
        super().__init__()
        if channels % 3 == 0:
            self.pre  = nn.Identity()
            self.post = nn.Identity()
            self.mab  = MAB(channels)
        else:
            new_c = max(12, channels // 4)  # 至少 12，确保能整除 3
            new_c -= new_c % 3
            # 向下取整
            if new_c < 3:
                raise ValueError(f'Channels={channels} too small for MAB')
            self.pre  = nn.Conv2d(channels, new_c, 1)
            self.mab  = MAB(new_c)
            self.post = nn.Conv2d(new_c, channels, 1)

    def forward(self, x):
        y = self.pre(x)
        y = self.mab(y)
        y = self.post(y)
        return x + y                           # 额外残差
