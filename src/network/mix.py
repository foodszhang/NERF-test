import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .unet import UNet
from .unet3 import UNet3Plus
from .point_classifier import SurfaceClassifier
from .network import DensityNetwork_debug
from src.encoder import get_encoder


def coord_to_dif(points):
    return ((points + 0.1275) / (0.1275 + 0.1275) * 2) - 1


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


class MixNet(nn.Module):
    def __init__(
        self, dif_net, nerf_net, hidden_dim=256, num_layers=8, skips=[4], out_dim=1
    ):
        self.layers = nn.ModuleList(
            [nn.Linear(2, hidden_dim)]
            + [
                (
                    nn.Linear(hidden_dim, hidden_dim)
                    if i not in skips
                    else nn.Linear(hidden_dim + self.in_dim, hidden_dim)
                )
                for i in range(1, num_layers - 1, 1)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.dif_net = dif_net
        self.nerf_net = nerf_net

        self.activations = nn.ModuleList(
            [nn.LeakyReLU() for i in range(0, num_layers - 1, 1)]
        )
        self.activations.append(nn.LeakyReLU())

    def forward(self, x):
        x1 = self.dif_net(x)
        x2 = self.nerf_net(x)
        x = torch.cat([x1, x2], -1)
        input_pts = x
        for i in range(len(self.layers)):

            linear = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = linear(x)
            x = activation(x)
        return x


class DIF_Net(nn.Module):
    def __init__(
        self,
        num_views,
        combine,
        mid_ch=128,
        image_encoding="unet3",
        position_encoding="hashgrid",
    ):
        super().__init__()
        self.combine = combine
        if image_encoding == "unet":
            self.image_encoding = "unet"
            self.image_encoder = UNet(1, mid_ch)
        else:
            self.image_encoding = "unet3"
            self.image_encoder = UNet3Plus(mid_ch, fast_up=False, use_cgm=False)
        self.position_encoder = get_encoder(position_encoding)
        # self.mlp = DensityNetwork_debug(mid_ch + 32)
        self.mlp = DensityNetwork_debug(mid_ch)

        if self.combine == "mlp":
            self.view_mixer = MLP([num_views, num_views // 2, 1])

        # self.point_classifier = SurfaceClassifier(
        #    [mid_ch + 32, 256, 64, 16, 1], no_residual=False
        # )
        self.point_classifier = SurfaceClassifier(
            [mid_ch, 256, 64, 16, 1], no_residual=False
        )
        print(f"DIF_Net, mid_ch: {mid_ch}, combine: {self.combine}")

    def forward(self, data, eval_npoint=102400):
        # projection encoding
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
            )  # B, C, N
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
        p_feats = torch.stack(p_list, dim=-1)  # B, C, N, M

        # 2. cross-view fusion
        if self.combine == "max":
            p_feats = F.max_pool2d(p_feats, (1, n_view))
            p_feats = p_feats.squeeze(-1)  # B, C, N
        elif self.combine == "mlp":
            p_feats = p_feats.permute(0, 3, 1, 2)
            p_feats = self.view_mixer(p_feats)
            p_feats = p_feats.squeeze(1)
        else:
            raise NotImplementedError

        # 3. point-wise classification
        # p_feats B, 128 , N
        # q = self.position_encoder(data["pts"], 0.2)  # B, N, 32
        # q = q.permute(0, 2, 1)
        # p_feats = torch.cat([p_feats, q], dim=1)

        # p_pred = self.point_classifier(p_feats)
        p_feats = p_feats.permute(0, 2, 1)
        p_pred = self.mlp(p_feats)
        p_pred = p_pred.permute(0, 2, 1)
        return p_pred
