import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import tinycudann as tcnn

from .unet import UNet
from .unet3 import UNet3Plus
from .point_classifier import SurfaceClassifier
from .network import DensityNetwork_debug
from src.encoder import get_encoder
from .cbp import CompactBilinearPooling


def coord_to_dif(points):
    return ((points + 0.1275) / (0.1275 + 0.1275) * 2) - 1


mlp_config = {
    "otype": "CutlassMLP",
    "activation": "LeakyReLU",
    "output_activation": "SoftPlus",
    "n_neurons": 256,
    "n_hidden_layers": 6,
}
encoding_config = {
    "otype": "Grid",
    "type": "Hash",
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "base_resolution": 16,
    "per_level_scale": 1.0,
    "interpolation": "Linear",
}


def index_2d(feat, uv):
    # https://zhuanlan.zhihu.com/p/137271718
    # feat: [B, C, H, W]
    # uv: [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    feat = feat.transpose(2, 3)  # [W, H]
    samples = torch.nn.functional.grid_sample(
        feat, uv, align_corners=True
    )  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]


class MLP(nn.Module):
    def __init__(self, mlp_list, use_bn=False):
        super().__init__()

        layers = []
        for i in range(len(mlp_list) - 1):
            layers += [nn.Conv2d(mlp_list[i], mlp_list[i + 1], kernel_size=1)]
            if use_bn:
                layers += [nn.BatchNorm2d(mlp_list[i + 1])]
            layers += [nn.LeakyReLU(inplace=True)]

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DIF_Net(nn.Module):
    def __init__(
        self,
        num_views,
        mid_ch=4,
        image_encoding="unet",
        position_encoding="hashgrid",
    ):
        super().__init__()
        if image_encoding == "unet":
            self.image_encoding = "unet"
            self.image_encoder = UNet(1, mid_ch)
        else:
            self.image_encoding = "unet3"
            self.image_encoder = UNet3Plus(mid_ch, fast_up=False, use_cgm=False)
        self.image_encoder.output_dim = mid_ch
        self.position_encoder = get_encoder(position_encoding)
        self.mlp = DensityNetwork_debug(mid_ch * num_views)

    def forward(self, data, eval_npoint=10240, with_feat=False):
        # projection encoding
        if with_feat:
            proj_feats = [data["proj_feats"]]
        else:
            projs = data["projections"]  # B, M, C, W, H
            b, m, w, h = projs.shape
            projs = projs.reshape(b * m, 1, w, h)  # B', C, W, H
            if self.training:
                if self.image_encoding == "unet3":
                    proj_feats = self.image_encoder(projs)["final_pred"]
                else:
                    proj_feats = self.image_encoder(projs)

            else:
                proj_feats = self.image_encoder(projs)

            proj_feats = list(proj_feats) if type(proj_feats) is tuple else [proj_feats]
            for i in range(len(proj_feats)):
                _, c_, w_, h_ = proj_feats[i].shape
                proj_feats[i] = proj_feats[i].reshape(b, m, c_, w_, h_)  # B, M, C, W, H

        # point-wise forward
        total_npoint = data["proj_pts"].shape[2]
        n_batch = int(np.ceil(total_npoint / eval_npoint))

        pred_list = []
        for i in range(n_batch):
            left = i * eval_npoint
            right = min((i + 1) * eval_npoint, total_npoint)
            p_pred = self.forward_points(
                proj_feats,
                {
                    "proj_pts": data["proj_pts"][..., left:right, :],
                    "pts": data["pts"][..., left:right, :],
                },
            )
            pred_list.append(p_pred)

        pred = torch.cat(pred_list, dim=2)
        return pred

    # points -> (10. 1024x10, 3)
    # proj -> (10, 1024x1, 2)
    def forward_points(self, proj_feats, data):
        n_view = proj_feats[0].shape[1]
        # 1. query view-specific features
        p_list = []
        for i in range(n_view):
            f_list = []
            for proj_f in proj_feats:
                feat = proj_f[:, i, ...]  # B, C, W, H

                p = data["proj_pts"][:, i, ...]  # B, N, 2
                p_feats = index_2d(feat, p)  # B, C, N
                f_list.append(p_feats)
            p_feats = torch.cat(f_list, dim=1)
            p_list.append(p_feats)
        p_feats = torch.cat(p_list, dim=1)  # B, C, N, M

        proj_feats = p_feats

        p_feats = p_feats.permute(0, 2, 1)
        p_pred = self.mlp(p_feats)
        p_pred = p_pred.permute(0, 2, 1)
        return p_pred


class NerfNetwork(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        encoding_config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 1.0,
            "interpolation": "Linear",
        }
        # self.encoding = tcnn.Encoding(3, encoding_config)
        self.encoding = get_encoder("frequency")
        # self.mlp = DensityNetwork_debug(32)

    def forward(self, x):
        # stx()
        """
        input: (N_rays x N_samples, 3)
        经过encoder后变成: (N_rays x N_samples, 32)
        """
        pts = x["pts"]
        b, n, c = pts.shape
        pts = pts.reshape(-1, c)
        x = self.encoding(pts)
        x = x.float()
        x = self.mlp(x)
        x = x.reshape(b, -1, 1)
        return x


class ImageNerfNetwork(nn.Module):
    def __init__(
        self,
        bound=0.4,
        num_layers=8,
        feat_dim=10 * 4,
        hidden_dim=256,
        skips=[4],
    ):
        super().__init__()
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.in_dim = feat_dim
        self.bound = bound

        self.encoding = get_encoder("hashgrid")
        # self.encoding = get_encoder("frequency")
        self.L = 16
        self.T = 400
        # self.encoding = tcnn.Encoding(3, encoding_config)
        # Linear layers
        self.feat_dim = feat_dim
        # self.total_dim = 128
        # self.total_dim = feat_dim + 32 + 10
        self.total_dim = feat_dim + 32
        # self.total_dim = 32 + 10
        ##self.total_dim = 32
        # self.mlp = DensityNetwork_debug(self.total_dim, num_layers=5, hidden_dim=128)
        # self.mlp = DensityNetwork_debug(self.total_dim, num_layers=5, hidden_dim=128)
        self.mlp = tcnn.Network(self.total_dim, 1, mlp_config)
        # self.feature_mix_layer = CompactBilinearPooling(
        #    self.feat_dim, 32, self.total_dim, sum_pool=False
        # )
        z_linears = []
        mlps = []
        net_width = 256
        self.first_layer = nn.Sequential(nn.Linear(32, net_width))
        for i in range(5):
            z_linears.append(nn.Linear(40, net_width))
            mlps.append(
                nn.Sequential(
                    nn.Linear(net_width, net_width),
                    nn.ReLU(),
                    nn.Linear(net_width, net_width),
                    nn.ReLU(),
                )
            )
        self.z_linears = nn.ModuleList(z_linears)
        self.mlps = nn.ModuleList(mlps)
        self.final_layer = nn.Linear(net_width, 1)

    def forward(self, x, t):
        # stx()
        """
        input: (N_rays x N_samples, 3)
        经过encoder后变成: (N_rays x N_samples, 32)
        """
        pts = x["pts"]
        proj_feats = x["projs_feats"]
        b, m, c, w, h = proj_feats.shape
        # 1, 10, 8 ,256, 256
        n_view = m
        # 1. query view-specific features
        p_list = []
        for i in range(n_view):
            f_list = []
            for j in range(c):
                feat = proj_feats[:, i, j, ...]  # B, C, W, H
                feat = feat.reshape(1, *feat.shape)
                p = x["proj_pts"][:, i, ...]  # B, N, 2
                p_feats = index_2d(feat, p)  # B, C, N
                f_list.append(p_feats)
            p_feats = torch.cat(f_list, dim=1)
            p_list.append(p_feats)
        p_feats = torch.cat(p_list, dim=1)  # B, C, N, M
        b, n, c = pts.shape
        pts = pts.reshape(-1, c)
        # p_feats = (p_feats - p_feats.min()) / (p_feats.max() - p_feats.min())
        pos_feats = self.encoding(pts, self.bound)
        # pos_feats = (pos_feats - pos_feats.min()) / (pos_feats.max() - pos_feats.min())
        pos_feats = pos_feats.float()
        pos_feats = pos_feats.reshape(b, n, -1)
        p_feats = p_feats.permute(0, 2, 1)
        # outputs = self.first_layer(pos_feats)
        # for idx in range(5):
        #    resnet_zs = self.z_linears[idx](p_feats)
        #    outputs = outputs + resnet_zs
        #    outputs = self.mlps[idx](outputs) + outputs

        # outputs = self.final_layer(outputs)
        # c = torch.relu(outputs)
        # return c
        # if "dif_out" in x:
        #    dif_out = x["dif_out"].permute(0, 2, 1)
        #    dif_out = torch.cat([dif_out for i in range(10)], dim=2)
        #    p_feats = torch.cat([pos_feats, p_feats, dif_out], dim=2)
        #    # p_feats = torch.cat([pos_feats, dif_out], dim=2)
        #    # p_feats = dif_out
        # else:
        #    p_feats = torch.cat([pos_feats, p_feats], dim=2)
        # pos_feats[int(t / self.T * self.L) * 2 + 2 :] = 0

        p_feats = torch.cat([pos_feats, p_feats], dim=2)
        # p_feats = pos_feats
        # p_feats = self.norm(p_feats)
        # p_feats = self.feature_mix_layer(
        #    p_feats,
        #    pos_feats,
        # )
        x = [self.mlp(p_feat) for p_feat in p_feats]
        x = torch.cat(x, dim=1)  # B, C, N, M
        # return outputs.view(b, -1)
        x = x.reshape(b, n, 1)
        return x
