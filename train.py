import os
import os.path as osp
import torch
import imageio.v2 as iio
import numpy as np
from tqdm import tqdm
import argparse
import skimage as ski


def coord_to_dif(points):
    return ((points + 0.1275) / (0.1275 + 0.1275) * 2) - 1


# TODO: HARD CODE
def index_3d(image, uv):
    # feat: [D, H, W]
    # uv: [N, 3]
    # uv = uv.reshape(1, *uv.shape) # [1, B, N, 3]
    image = image.unsqueeze(0)  # [1, D, H, W]
    image = image.unsqueeze(0)  # [1, D, H, W]
    uv = uv.unsqueeze(0)  # [B, N, 1, 3]
    uv = uv.unsqueeze(2)  # [B, N, 1, 3]
    uv = uv.unsqueeze(2)  # [B, N, 1, 3]
    uv = coord_to_dif(uv)  # [B, N, 1, 3]
    # image = image.transpose(2, 3) # [W, H]
    samples = torch.nn.functional.grid_sample(
        image, uv, align_corners=True
    )  # [B, C, N, 1]
    return samples[0, 0, :, :, 0]  # [B, C, N]


def config_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default=f"./config/nerf/chest_50.yaml", help="configs file path")
    parser.add_argument(
        "--config",
        default=f"./config/Lineformer/luna16_50_simple.yaml",
        help="configs file path",
    )
    parser.add_argument("--gpu_id", default="0", help="gpu to use")
    return parser


parser = config_parser()
args = parser.parse_args()
print("!!!!!!", torch.cuda.is_available())

# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
# os.environ["CUDA_HOME"]='C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.3'

from src.config.configloading import load_config
from src.render import render, run_network
from src.trainer import Trainer
from src.loss import calc_mse_loss, calc_tv_loss, compute_tv_norm
from src.utils import get_psnr, get_ssim, get_psnr_3d, get_ssim_3d, cast_to_image
from pdb import set_trace as stx


cfg = load_config(args.config)


# torch.cuda.set_device(2)

# stx()
device = torch.device("cuda")
# stx()


# 从Trainer继承
class BasicTrainer(Trainer):
    def __init__(self):
        """
        Basic network trainer.
        """
        super().__init__(cfg, "cuda")
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")

    def compute_loss(self, data, global_step, idx_epoch):
        # stx()
        rays = data["rays"].reshape(-1, 8)  # [1, 1024, 8] -> [1024, 8]

        projs = data["projs"].reshape(
            -1
        )  # projection 的 ground truth [1, 1024] -> [1024]
        ret = render(rays, self.net, self.net_fine, **self.conf["render"])
        # stx()
        projs_pred = ret["acc"]
        loss = {"loss": 0.0}
        calc_mse_loss(loss, projs, projs_pred)
        # Log
        for ls in loss.keys():
            self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)

        return loss["loss"]

    def sample_points(self, points):
        choice = np.random.choice(len(points), size=50000, replace=False)
        points = points[choice]
        return points

    # def compute_loss(self, data, global_step, idx_epoch):
    #    points = self.eval_dset.voxels.reshape(-1, 3)
    #    points = self.sample_points(points)

    #    image_pred = run_network(
    #        points, self.net_fine if self.net_fine is not None else self.net, self.netchunk)
    #    #image_pred = image_pred.squeeze()
    #    image = index_3d(self.net.pre_image, points)
    #    loss = {"loss": 0.}
    #    calc_mse_loss(loss, image_pred, image)
    #    for ls in loss.keys():
    #        self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)

    #    return loss["loss"]

    #    # stx()

    def eval_step(self, global_step, idx_epoch):
        """
        Evaluation step
        """
        # Evaluate projection    渲染投射的 RGB 图
        projs = self.eval_dset.projs  # [256, 256] -> [50, 256, 256]
        rays = self.eval_dset.rays.reshape(-1, 8)  # [65536,8]  -> [3276800, 8]
        # stx()
        N, H, W = projs.shape
        # projs_pred = []
        # for i in tqdm(range(0, rays.shape[0], self.n_rays)):     # 每一簇射线是 n_rays ，每隔这么多射线渲染一次
        #    projs_pred.append(render(rays[i:i+self.n_rays], self.net, self.dif_net, self.net_fine, **self.conf["render"])["acc"])
        # projs_pred = torch.cat(projs_pred, 0).reshape(N, H, W)

        # Evaluate density      渲染3D图像
        image = self.eval_dset.image
        image_pred = run_network(
            self.eval_dset.voxels,
            self.net_fine if self.net_fine is not None else self.net,
            self.netchunk,
        )
        # stx()
        image_pred = image_pred.squeeze()
        # stx()
        loss = {
            # "proj_psnr": get_psnr(projs_pred, projs),
            # "proj_ssim": get_ssim(projs_pred, projs),
            "psnr_3d": get_psnr_3d(image_pred, image),
            "ssim_3d": get_ssim_3d(image_pred, image),
        }
        if loss["ssim_3d"] > self.best_ssim_3d:
            torch.save(
                {
                    "epoch": idx_epoch,
                    "network": self.net.state_dict(),
                    "network_fine": (
                        self.net_fine.state_dict() if self.n_fine > 0 else None
                    ),
                    "optimizer": self.optimizer.state_dict(),
                },
                self.ckpt_best_dir,
            )
            self.best_ssim_3d = loss["ssim_3d"]
            self.logger.info(
                f"best model update, epoch:{idx_epoch}, best 3d psnr:{self.best_ssim_3d:.4g}"
            )

        # stx()

        # Logging
        show_slice = 5
        show_step = image.shape[-1] // show_slice
        show_image = image[..., ::show_step]
        show_image_pred = image_pred[..., ::show_step]
        show = []
        for i_show in range(show_slice):
            show.append(
                torch.concat(
                    [show_image[..., i_show], show_image_pred[..., i_show]], dim=0
                )
            )
        show_density = torch.concat(show, dim=1)

        # cast_to_image -> 转成 numpy并多加一个维度
        # self.writer.add_image("eval/density (row1: gt, row2: pred)", cast_to_image(show_density), global_step, dataformats="HWC")

        # proj_pred_origin_dir = osp.join(self.expdir, "proj_pred_origin")
        # proj_gt_origin_dir = osp.join(self.expdir, "proj_gt_origin")
        # proj_pred_dir = osp.join(self.expdir, "proj_pred")
        # proj_gt_dir = osp.join(self.expdir, "proj_gt")
        ## os.makedirs(eval_save_dir, exist_ok=True)
        # os.makedirs(proj_pred_origin_dir, exist_ok=True)
        # os.makedirs(proj_gt_origin_dir, exist_ok=True)
        # os.makedirs(proj_pred_dir, exist_ok=True)
        # os.makedirs(proj_gt_dir, exist_ok=True)

        # for i in tqdm(range(N)):
        #    '''
        #        cast_to_image 自带了归一化, 1 - 放在外边
        #    '''
        #    ski.io.imsave(osp.join(proj_pred_origin_dir, f"proj_pred_{str(i)}.png"), ski.color.gray2rgb((cast_to_image(projs_pred[i])*255).astype(np.uint8)))
        #    ski.io.imsave(osp.join(proj_gt_origin_dir, f"proj_gt_{str(i)}.png"), ski.color.gray2rgb((cast_to_image(projs[i])*255).astype(np.uint8)))
        #    ski.io.imsave(osp.join(proj_pred_dir, f"proj_pred_{str(i)}.png"), ski.color.gray2rgb(((1-cast_to_image(projs_pred[i]))*255).astype(np.uint8)))
        #    ski.io.imsave(osp.join(proj_gt_dir, f"proj_gt_{str(i)}.png"), ski.color.gray2rgb(((1-cast_to_image(1-projs[i]))*255).astype(np.uint8)))

        # stx()
        for ls in loss.keys():
            self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)

        # Save
        # 保存各种视图
        eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        np.save(
            osp.join(eval_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy()
        )
        np.save(osp.join(eval_save_dir, "image_gt.npy"), image.cpu().detach().numpy())
        iio.imwrite(
            osp.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"),
            ski.color.gray2rgb((cast_to_image(show_density) * 255).astype(np.uint8)),
        )
        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f:
            for key, value in loss.items():
                f.write("%s: %f\n" % (key, value.item()))

        return loss


trainer = BasicTrainer()
# 这并不是多线程中的start函数，而是父类Trainer中的start函数
trainer.start()  # loop train and evaluation
