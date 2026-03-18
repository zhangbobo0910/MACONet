import numpy
from torch import nn
import torch

import torch.nn.functional as F

from .cnn_liabrary import *


# 哔哩哔哩：CV缝合救星
class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels,
                                   bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


def get_norm_layer(norm_type, channels, num_groups):
    """获取归一化层，自动处理GroupNorm的分组数不能整除的问题"""
    if norm_type == 'GN':
        # 确保num_groups能整除channels
        if channels % num_groups != 0:
            # 选择最大的约数 <= num_groups，且能整除channels
            possible_groups = [g for g in range(1, num_groups + 1) if channels % g == 0]
            if not possible_groups:
                # 如果没有合适的约数，使用1
                num_groups = 1
            else:
                num_groups = max(possible_groups)
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm3d(channels)


class GBConv(nn.Module):
    def __init__(self, in_channels, norm_type='GN'):
        super(GBConv, self).__init__()

        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        x = x1 * x2
        x = self.block4(x)

        return x + residual


class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        """

        :param x: bsz x dim x max_len x max_len
        :param mask: bsz x dim x max_len x max_len, 为1的地方为pad
        :return:
        """
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


# 原BFM卷积块定义（对应论文公式8）
class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.ln = nn.LayerNorm(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.ln(self.conv(x)))


# 替换为GBConv的BFM卷积块
class BFM_GBConv(nn.Module):
    def __init__(self, channels, norm_type='GN'):
        super().__init__()
        # 复用GBConv的多分支结构，匹配BFM的多尺度特征需求
        self.gbconv = GBConv(in_channels=channels, norm_type=norm_type)
        # 保留论文中的下采样操作（用于自上而下路径）
        self.downsample = nn.Conv2d(channels, channels, 3, padding=1, stride=2)

    def forward(self, x):
        x = self.gbconv(x)  # 替代原ConvBlock的特征细化
        x_down = self.downsample(x)  # 保持下采样逻辑
        return x, x_down


class BoundaryFiltrationModule(nn.Module):
    def __init__(self, channels, n_blocks=2):  # n_blocks对应论文中的n+1=2
        super().__init__()
        self.blocks = nn.ModuleList([BFM_GBConv(channels) for _ in range(n_blocks)])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 论文中的nearest上采样

    def forward(self, x):
        # 自上而下路径：提取语义交互
        down_feats = []
        for block in self.blocks:
            x, x_down = block(x)
            down_feats.append(x)  # 保存中间特征用于横向连接
            x = x_down

        # 自下而上路径：恢复细节（对应论文公式9）
        for i in reversed(range(len(down_feats) - 1)):
            x = self.upsample(x) + down_feats[i]  # 横向连接融合

        return x


class MaskCNN_1(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, depth=3, theta=1):
        super(MaskCNN_1, self).__init__()
        self.depth = depth
        self.theta = theta
        layers1 = []
        layers1.append(Conv2d_selfAdapt(input_channels, input_channels, kernel_size, padding=kernel_size // 2))
        layers1.append(LayerNorm((1, input_channels, 1, 1), dim_index=1))
        layers1.append(nn.GELU())
        self.layer1 = nn.Sequential(*layers1)

        layers2 = []
        layers2.append(Conv2d_selfAdapt(input_channels, input_channels, kernel_size, padding=kernel_size // 2))
        layers2.append(LayerNorm((1, input_channels, 1, 1), dim_index=1))
        layers2.append(nn.GELU())
        self.layer2 = nn.Sequential(*layers2)

        layers3 = []
        layers3.append(Conv2d_selfAdapt(input_channels, input_channels, kernel_size, padding=kernel_size // 2))
        layers3.append(LayerNorm((1, input_channels, 1, 1), dim_index=1))
        layers3.append(nn.GELU())
        self.layer3 = nn.Sequential(*layers3)

        layers4 = []
        layers4.append(Conv2d_selfAdapt(input_channels, input_channels, kernel_size, padding=kernel_size // 2))
        layers4.append(LayerNorm((1, input_channels, 1, 1), dim_index=1))
        layers4.append(nn.GELU())
        # 在layers4中替换原有卷积块为GBConv
        layers4.extend([
            GBConv(input_channels),  # 替换原有MaskConv2d+LayerNorm+GELU
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, stride=2),  # 保持下采样
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(),
        ])
        self.layer4 = nn.Sequential(*layers4)

        self.attn_dc1 = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.attn_dc2 = nn.Conv2d(input_channels, 1, kernel_size=1)

        # 添加GBConv模块（输入通道数为input_channels，与SDM输出一致）
        self.gbconv = GBConv(input_channels)  # 复用B站的GBConv实现

        # 添加BoundaryFiltrationModule
        self.bfm = BoundaryFiltrationModule(input_channels)

    def forward(self, x, mask, train):
        """

        :param x: bsz x input_channels x max_len x max_len
        :param mask: bsz x max_len x max_len, 为1的地方是有span
        :return:
        """
        bsz, input_channels, max_len, _ = x.size()
        mask_2d = mask[:, None, :, :].eq(0)

        # 修复：在调用Conv2d_selfAdapt时传递init_flag参数
        x_1 = self.layer1[0](x, init_flag=train)
        x_1 = self.layer1[1](x_1)
        x_1 = self.layer1[2](x_1)
        x_1 = x_1.masked_fill(mask_2d, 0)

        x_2 = self.layer2[0](x_1, init_flag=train)
        x_2 = self.layer2[1](x_2)
        x_2 = self.layer2[2](x_2)
        x_2 = x_2.masked_fill(mask_2d, 0)

        x_3 = self.layer3[0](x_2, init_flag=train)
        x_3 = self.layer3[1](x_3)
        x_3 = self.layer3[2](x_3)
        x_3 = x_3.masked_fill(mask_2d, 0)

        x_4 = self.layer4[0](x_3, init_flag=train)
        x_4 = self.layer4[1](x_4)
        x_4 = self.layer4[2](x_4)
        x_4 = self.layer4[3](x_4)  # GBConv
        x_4 = self.layer4[4](x_4)  # Conv2d下采样
        x_4 = self.layer4[5](x_4)
        x_4 = self.layer4[6](x_4)
        x_4 = x_4.masked_fill(mask_2d, 0)

        # 在SDM的SAD块输出后应用GBConv
        x_1 = self.gbconv(x_1)  # 增强差分特征
        x_2 = self.gbconv(x_2)
        x_3 = self.gbconv(x_3)

        # 应用BoundaryFiltrationModule
        x_bfm = self.bfm(x_4)

        # 后续融合逻辑不变...
        attn_dc1 = self.attn_dc1(x_1).squeeze(1)
        attn_dc2 = self.attn_dc2(x_2).squeeze(1)

        # 应用mask
        attn_dc1 = attn_dc1.masked_fill(mask.eq(0), -10000)
        attn_dc2 = attn_dc2.masked_fill(mask.eq(0), -10000)

        # softmax
        attn_dc1 = F.softmax(attn_dc1.view(bsz, -1), dim=-1).view(bsz, max_len, max_len)
        attn_dc2 = F.softmax(attn_dc2.view(bsz, -1), dim=-1).view(bsz, max_len, max_len)

        # 加权平均
        x_1 = torch.einsum('bmn,bcmn->bcmn', attn_dc1, x_1)
        x_2 = torch.einsum('bmn,bcmn->bcmn', attn_dc2, x_2)

        # 结合BFM的输出
        x_bfm_expanded = F.interpolate(x_bfm, size=(max_len, max_len), mode='bilinear', align_corners=False)
        x_bfm_expanded = x_bfm_expanded.masked_fill(mask_2d, 0)

        # 最终输出
        linear_atts_dc1 = x_1 + x_2 + x_bfm_expanded

        return linear_atts_dc1


class MaskCNN_2(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, depth=3, theta=1):
        super(MaskCNN_2, self).__init__()
        self.depth = depth
        self.theta = theta
        layers1 = []
        layers1.append(Conv2d_selfAdapt(input_channels, input_channels, kernel_size, padding=kernel_size // 2))
        layers1.append(LayerNorm((1, input_channels, 1, 1), dim_index=1))
        layers1.append(nn.GELU())
        self.layer1 = nn.Sequential(*layers1)

        layers2 = []
        layers2.append(Conv2d_selfAdapt(input_channels, input_channels, kernel_size, padding=kernel_size // 2))
        layers2.append(LayerNorm((1, input_channels, 1, 1), dim_index=1))
        layers2.append(nn.GELU())
        self.layer2 = nn.Sequential(*layers2)

        layers3 = []
        layers3.append(Conv2d_selfAdapt(input_channels, input_channels, kernel_size, padding=kernel_size // 2))
        layers3.append(LayerNorm((1, input_channels, 1, 1), dim_index=1))
        layers3.append(nn.GELU())
        self.layer3 = nn.Sequential(*layers3)

        layers4 = []
        layers4.append(Conv2d_selfAdapt(input_channels, input_channels, kernel_size, padding=kernel_size // 2))
        layers4.append(LayerNorm((1, input_channels, 1, 1), dim_index=1))
        layers4.append(nn.GELU())
        # 在layers4中替换原有卷积块为GBConv
        layers4.extend([
            GBConv(input_channels),  # 替换原有MaskConv2d+LayerNorm+GELU
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, stride=2),  # 保持下采样
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(),
        ])
        self.layer4 = nn.Sequential(*layers4)

        self.attn_dc1 = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.attn_dc2 = nn.Conv2d(input_channels, 1, kernel_size=1)

        # 添加GBConv模块
        self.gbconv = GBConv(input_channels)

        # 添加BoundaryFiltrationModule
        self.bfm = BoundaryFiltrationModule(input_channels)

    def forward(self, x, mask, train):
        """

        :param x: bsz x input_channels x max_len x max_len
        :param mask: bsz x max_len x max_len, 为1的地方是有span
        :return:
        """
        bsz, input_channels, max_len, _ = x.size()
        mask_2d = mask[:, None, :, :].eq(0)

        # 修复：在调用Conv2d_selfAdapt时传递init_flag参数
        x_1 = self.layer1[0](x, init_flag=train)
        x_1 = self.layer1[1](x_1)
        x_1 = self.layer1[2](x_1)
        x_1 = x_1.masked_fill(mask_2d, 0)

        x_2 = self.layer2[0](x_1, init_flag=train)
        x_2 = self.layer2[1](x_2)
        x_2 = self.layer2[2](x_2)
        x_2 = x_2.masked_fill(mask_2d, 0)

        x_3 = self.layer3[0](x_2, init_flag=train)
        x_3 = self.layer3[1](x_3)
        x_3 = self.layer3[2](x_3)
        x_3 = x_3.masked_fill(mask_2d, 0)

        x_4 = self.layer4[0](x_3, init_flag=train)
        x_4 = self.layer4[1](x_4)
        x_4 = self.layer4[2](x_4)
        x_4 = self.layer4[3](x_4)  # GBConv
        x_4 = self.layer4[4](x_4)  # Conv2d下采样
        x_4 = self.layer4[5](x_4)
        x_4 = self.layer4[6](x_4)
        x_4 = x_4.masked_fill(mask_2d, 0)

        # 应用GBConv
        x_1 = self.gbconv(x_1)
        x_2 = self.gbconv(x_2)
        x_3 = self.gbconv(x_3)

        # 应用BoundaryFiltrationModule
        x_bfm = self.bfm(x_4)

        attn_dc1 = self.attn_dc1(x_1).squeeze(1)
        attn_dc2 = self.attn_dc2(x_2).squeeze(1)

        # 应用mask
        attn_dc1 = attn_dc1.masked_fill(mask.eq(0), -10000)
        attn_dc2 = attn_dc2.masked_fill(mask.eq(0), -10000)

        # softmax
        attn_dc1 = F.softmax(attn_dc1.view(bsz, -1), dim=-1).view(bsz, max_len, max_len)
        attn_dc2 = F.softmax(attn_dc2.view(bsz, -1), dim=-1).view(bsz, max_len, max_len)

        # 加权平均
        x_1 = torch.einsum('bmn,bcmn->bcmn', attn_dc1, x_1)
        x_2 = torch.einsum('bmn,bcmn->bcmn', attn_dc2, x_2)

        # 结合BFM的输出
        x_bfm_expanded = F.interpolate(x_bfm, size=(max_len, max_len), mode='bilinear', align_corners=False)
        x_bfm_expanded = x_bfm_expanded.masked_fill(mask_2d, 0)

        # 最终输出
        linear_atts_dc1 = x_1 + x_2 + x_bfm_expanded

        return linear_atts_dc1