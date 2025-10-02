import os
import math
import torch
import warnings
import torch.nn as nn
from thop import profile
from functools import partial
import torch.nn.functional as F
from lib.DPGNet.pvtv2 import pvt_v2_b2
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Head(nn.Module):
    def __init__(self, in_chans=64, out_chans=64):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_chans, out_chans, kernel_size=1)

    def forward(self, x):
        x = self.conv1x1(x)
        return x


class UpSamplingBlock(nn.Module):
    def __init__(self, up_size=2,
                 in_chans=64, out_chans=64):
        super().__init__()
        self.upSample = nn.UpsamplingBilinear2d(scale_factor=up_size)
        self.Conv1x1 = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        self.norm = nn.LayerNorm(out_chans)

    def forward(self, x):
        x = self.Conv1x1(self.upSample(x))
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


class DownPoolingBlock(nn.Module):
    def __init__(self, down_size=2,
                 in_chans=64, out_chans=64):
        super().__init__()
        self.pooling = nn.AvgPool2d(down_size)
        self.Conv1x1 = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        self.norm = nn.LayerNorm(out_chans)

    def forward(self, x):
        x = self.Conv1x1(self.pooling(x))
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


class ChannelAdjustBlock(nn.Module):
    def __init__(self, in_chans=64, out_chans=64):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        self.norm = nn.LayerNorm(out_chans)

    def forward(self, x):
        x = self.Conv1x1(x)
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ResMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.fc_res = nn.Linear(in_features, out_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x_res = self.fc_res(x)
        x_res = self.act(x_res)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x * x_res
        return x


class PoolAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.AvgPool2d(kernel_size=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PoolAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.res_mlp = ResMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.res_mlp(self.norm2(x)))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


class CrossDotAttention(nn.Module):
    def __init__(self, feature_chans=64):
        super().__init__()

    def forward(self, x_q, x_k, x_v):
        x_out = torch.sigmoid(x_q * x_k) * x_v

        return x_out


class MinMaxScaling(nn.Module):
    def __init__(self, feature_chans=64):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2)
        x_minmax = min_max_norm(x, dim=2)
        x_minmax = x_minmax.reshape(B, C, H, W)

        return x_minmax


class DualEdgeAwareness(nn.Module):
    def __init__(self, feature_chans=64):
        super().__init__()
        self.MMS = MinMaxScaling(feature_chans=feature_chans)

    def forward(self, x_edg, x_up, x_down):
        x_dea = x_edg * (1. - self.MMS(x_up)) * self.MMS(x_down)
        x_out = x_dea

        return x_out


def min_max_norm(x, dim=2):
    x = torch.moveaxis(x, dim, -1)

    min = torch.min(x, dim=-1, keepdim=True)[0]
    max = torch.max(x, dim=-1, keepdim=True)[0]
    x = x - min

    minmax = x / (max - min + 1e-8)

    out = torch.moveaxis(minmax, -1, dim)

    return out


class FeatureScaleAlignment(nn.Module):
    def __init__(self, feature_chans=64):
        super().__init__()
        self.MMS = MinMaxScaling()

    def forward(self, x_up, x_down):
        x_up = self.MMS(x_up)
        x_down = self.MMS(x_down)
        x_diff = x_up + x_down
        x_diff = self.MMS(x_diff)
        return x_diff


class FeatureDifferenceEnhancement(nn.Module):
    def __init__(self, feature_chans=64):
        super().__init__()
        self.MMS = MinMaxScaling()

    def forward(self, x_up, x_down):
        x_up = self.MMS(x_up)
        x_down = self.MMS(x_down)
        x_diff = x_up - x_down
        x_diff = self.MMS(x_diff)
        return x_diff


class Neck(nn.Module):
    def __init__(self, in_chans=[64, 128, 320, 512], out_chans=64):
        super().__init__()
        num_stages = len(in_chans)
        self.chans_cut = nn.ModuleList(
            [ChannelAdjustBlock(in_chans=in_chans[i], out_chans=out_chans)
             for i in range(num_stages)])

    def forward(self, x4, x3, x2, x1):
        x_list = [x4, x3, x2, x1]
        for i in range(len(x_list)):
            x_list[i] = self.chans_cut[i](x_list[i])
        return x_list


# decoder block
class TransformerMultiFusion(nn.Module):
    def __init__(self, feature_chans=64):
        super().__init__()

        # self CDA
        self.CDA = CrossDotAttention(feature_chans=feature_chans)

        # up sampling
        self.USB1 = UpSamplingBlock(up_size=8, in_chans=feature_chans, out_chans=feature_chans)
        self.USB2 = UpSamplingBlock(up_size=4, in_chans=feature_chans, out_chans=feature_chans)
        self.USB3 = UpSamplingBlock(up_size=2, in_chans=feature_chans, out_chans=feature_chans)
        self.USB4 = UpSamplingBlock(up_size=1, in_chans=feature_chans, out_chans=feature_chans)

        # multi fusion
        self.CAB = ChannelAdjustBlock(in_chans=feature_chans * 4, out_chans=feature_chans)

        # transformer pool self attn
        self.PABlocks = nn.ModuleList([PoolAttentionBlock(dim=feature_chans, num_heads=4, mlp_ratio=4., sr_ratio=8,
                                                          ) for i in range(2)])

    def forward(self, x1, x2, x3, x4):
        x11 = x1
        x22 = x2
        x44 = x3
        x88 = x4

        x11_cda = self.CDA(x11, x11, x11)
        x22_cda = self.CDA(x22, x22, x22)
        x44_cda = self.CDA(x44, x44, x44)
        x88_cda = self.CDA(x88, x88, x88)

        x11_up = self.USB1(x11_cda)
        x22_up = self.USB2(x22_cda)
        x44_up = self.USB3(x44_cda)
        x88_up = self.USB4(x88_cda)

        x_fusion = self.CAB(torch.concat([x11_up, x22_up, x44_up, x88_up], dim=1))

        x_sa = x_fusion

        for i in range(2):
            x_sa = self.PABlocks[i](x_sa)

        x_out = x_sa

        return x_out


class DownUpCommunication(nn.Module):
    def __init__(self, feature_chans=64, sr_ratio=8, up_size=2):
        super().__init__()

        # USB
        self.USB = UpSamplingBlock(up_size=up_size, in_chans=feature_chans, out_chans=feature_chans)

        # CDA
        self.CDA = CrossDotAttention(feature_chans=feature_chans)

        # pool PVT Transformer
        self.PABlock = PoolAttentionBlock(dim=feature_chans, num_heads=4, mlp_ratio=4., sr_ratio=sr_ratio)

        # FSA
        self.FSA = FeatureScaleAlignment(feature_chans=feature_chans)

    def forward(self, x_pre, x_current, x_edg, x_down):
        x_pre = self.USB(x_pre)
        x_cross1 = self.CDA(x_pre, x_edg, x_edg)
        x_cross2 = self.CDA(x_current, x_edg, x_edg)
        x_cross3 = self.CDA(x_current, x_cross1, x_cross1)
        x_cross = x_cross1 + x_cross2 + x_cross3
        x_sa = self.PABlock(x_cross)
        x_fsa = self.FSA(x_sa, x_down)

        x_out = x_fsa

        return x_out


class EdgeDifferenceAttention(nn.Module):
    def __init__(self, feature_chans=64, sr_ratio=8):
        super().__init__()
        # FDE
        self.FDE = FeatureDifferenceEnhancement(feature_chans=feature_chans)

        # pool PVT Transformer
        self.PABlock = PoolAttentionBlock(dim=feature_chans, num_heads=4, mlp_ratio=4., sr_ratio=sr_ratio)

        # CDA
        self.CDA = CrossDotAttention(feature_chans=feature_chans)

        # FSA
        self.FSA = FeatureScaleAlignment(feature_chans=feature_chans)

        # DEA
        self.DEA = DualEdgeAwareness(feature_chans=feature_chans)

    def forward(self, x_up, x_down):
        x_fde = self.FDE(x_up, x_down)
        x_dea = self.DEA(x_fde, x_up, x_down)
        x_sa = self.PABlock(x_dea)
        x_cda = self.CDA(x_fde, x_sa, x_sa)
        x_edg = self.FSA(x_fde, x_dea)
        x_fsa = self.FSA(x_cda, x_edg)

        x_out = x_fsa

        return x_out


class Decoder(nn.Module):
    # pyramid vision transformer decoder
    def __init__(self, feature_chans=64):
        super().__init__()

        # TMF
        self.TMF = TransformerMultiFusion(feature_chans=feature_chans)

        # DPB
        self.DPB1 = DownPoolingBlock(down_size=8, in_chans=feature_chans, out_chans=feature_chans)
        self.DPB2 = DownPoolingBlock(down_size=8, in_chans=feature_chans, out_chans=feature_chans)
        self.DPB3 = DownPoolingBlock(down_size=4, in_chans=feature_chans, out_chans=feature_chans)
        self.DPB4 = DownPoolingBlock(down_size=2, in_chans=feature_chans, out_chans=feature_chans)
        self.DPB5 = DownPoolingBlock(down_size=1, in_chans=feature_chans, out_chans=feature_chans)

        # USB
        self.USB1 = UpSamplingBlock(up_size=2, in_chans=feature_chans, out_chans=feature_chans)
        self.USB2 = UpSamplingBlock(up_size=2, in_chans=feature_chans, out_chans=feature_chans)
        self.USB3 = UpSamplingBlock(up_size=2, in_chans=feature_chans, out_chans=feature_chans)

        # DUC
        self.DUC1 = DownUpCommunication(feature_chans=feature_chans, sr_ratio=1, up_size=1)
        self.DUC2 = DownUpCommunication(feature_chans=feature_chans, sr_ratio=2, up_size=2)
        self.DUC3 = DownUpCommunication(feature_chans=feature_chans, sr_ratio=4, up_size=2)
        self.DUC4 = DownUpCommunication(feature_chans=feature_chans, sr_ratio=8, up_size=2)

        # EDA
        self.EDA1 = EdgeDifferenceAttention(feature_chans=feature_chans, sr_ratio=2)
        self.EDA2 = EdgeDifferenceAttention(feature_chans=feature_chans, sr_ratio=4)
        self.EDA3 = EdgeDifferenceAttention(feature_chans=feature_chans, sr_ratio=8)

        # head
        self.head1 = Head(in_chans=feature_chans, out_chans=1)
        self.head2 = Head(in_chans=feature_chans, out_chans=1)
        self.head3 = Head(in_chans=feature_chans, out_chans=1)
        self.head4 = Head(in_chans=feature_chans, out_chans=1)
        self.head5 = Head(in_chans=feature_chans, out_chans=1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_decoder(self, x4, x3, x2, x1):
        x_tmf = self.TMF(x1, x2, x3, x4)

        # stage 1
        x1_pre = x1
        x1_current = x1
        x1_up = self.DPB1(x_tmf)
        x1_down = self.DPB2(x_tmf)
        x1_duc = self.DUC1(x1_pre, x1_current, x1_up, x1_down)

        # stage 2
        x2_pre = x1
        x2_current = x2
        x2_up = self.USB1(x1_duc)
        x2_down = self.DPB3(x_tmf)
        x2_eda = self.EDA1(x2_up, x2_down)
        x2_duc = self.DUC2(x2_pre, x2_current, x2_eda, x2_down)

        # stage 3
        x3_pre = x2
        x3_current = x3
        x3_up = self.USB2(x2_duc)
        x3_down = self.DPB4(x_tmf)
        x3_eda = self.EDA2(x3_up, x3_down)
        x3_duc = self.DUC3(x3_pre, x3_current, x3_eda, x3_down)

        # stage 4
        x4_pre = x3
        x4_current = x4
        x4_up = self.USB3(x3_duc)
        x4_down = self.DPB5(x_tmf)
        x4_eda = self.EDA3(x4_up, x4_down)
        x4_duc = self.DUC4(x4_pre, x4_current, x4_eda, x4_down)

        # pred head
        x_pred1 = self.head1(x_tmf)
        x_pred2 = self.head2(x1_duc)
        x_pred3 = self.head3(x2_duc)
        x_pred4 = self.head4(x3_duc)
        x_pred5 = self.head5(x4_duc)

        return x_pred1, x_pred2, x_pred3, x_pred4, x_pred5

    def forward(self, x4, x3, x2, x1):
        x = self.forward_decoder(x4, x3, x2, x1)
        return x


class DPGNet(nn.Module):
    """
    Unet shape PVTnet Transformer Multiscale Fusion use channels cut in decoder with up sample deep supervise
    """

    def __init__(self, pth_path='./pretrained_pth/pvt_v2_b2.pth'):
        super().__init__()

        # encoder
        self.encoder = pvt_v2_b2()  # [64, 128, 320, 512]

        # preload pvt pth
        self.pth_path = pth_path
        save_model = torch.load(pth_path)
        model_dict = self.encoder.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.encoder.load_state_dict(model_dict)

        # neck
        self.neck = Neck(out_chans=64)

        # decoder
        self.decoder = Decoder(feature_chans=64)

    def forward(self, x):
        x4, x3, x2, x1 = self.encoder(x)
        x4, x3, x2, x1 = self.neck(x4, x3, x2, x1)
        x_out1, x_out2, x_out3, x_out4, x_out5 = self.decoder(x4, x3, x2, x1)
        x_pred1 = F.interpolate(x_out1, scale_factor=4, mode='bilinear')
        x_pred2 = F.interpolate(x_out2, scale_factor=32, mode='bilinear')
        x_pred3 = F.interpolate(x_out3, scale_factor=16, mode='bilinear')
        x_pred4 = F.interpolate(x_out4, scale_factor=8, mode='bilinear')
        x_pred5 = F.interpolate(x_out5, scale_factor=4, mode='bilinear')

        return x_pred1, x_pred2, x_pred3, x_pred4, x_pred5


if __name__ == "__main__":
    X = torch.randn(1, 3, 352, 352)
    model = DPGNet(pth_path='../pretrained_pth/pvt_v2_b2.pth')
    Y = model(X)
    flops, params = profile(model, inputs=(X,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))
