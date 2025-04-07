import os
import os.path as osp
import torch
import imageio.v2 as iio
import numpy as np
from tqdm import tqdm
import argparse
import SimpleITK as sitk


def config_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default=f"./config/nerf/chest_50.yaml", help="configs file path")
    parser.add_argument(
        "--config",
        default=f"./config/Lineformer/luna16_50.yaml",
        help="configs file path",
    )
    parser.add_argument("--gpu_id", default="0", help="gpu to use")
    return parser


parser = config_parser()
args = parser.parse_args()
print("!!!!!!", torch.cuda.is_available)

# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

from src.config.configloading import load_config
from src.multi_trainer import Trainer
from src.loss import calc_mse_loss
from src.utils import get_psnr, get_ssim, get_psnr_3d, get_ssim_3d, cast_to_image
from pdb import set_trace as stx


cfg = load_config(args.config)


# torch.cuda.set_device(2)

# stx()
device = torch.device("cuda")
# stx()


def save_nifti(image, path):
    out = sitk.GetImageFromArray(image)
    sitk.WriteImage(out, path)


# 从Trainer继承
class BasicTrainer(Trainer):
    def __init__(self):
        """
        Basic network trainer.
        """
        super().__init__(cfg, device)
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")

    def compute_loss(self, data, global_step, idx_epoch):
        # stx()
        loss = {"loss": 0.0}
        # stx()
        image_pred = self.net(data)
        image = data["image"]
        image = image.reshape(-1)
        image_pred = image_pred.reshape(-1)
        calc_mse_loss(loss, image, image_pred)
        return loss["loss"]

    def eval_step(self, global_step, idx_epoch):
        """
        Evaluation step
        """
        # Evaluate projection    渲染投射的 RGB 图
        loss = {
            # "proj_psnr": 0.0,
            # "proj_ssim": get_ssim(projs_pred, projs),
            "psnr_3d": 0.0,
            "ssim_3d": 0.0,
        }
        for index, data in enumerate(self.eval_dset):
            # stx()
            image_pred = self.net(data)
            image = data["image"]
            image = image.reshape(256, 256, 256)
            image_pred = image_pred.reshape(256, 256, 256)
            # stx()
            loss["ssim_3d"] += get_ssim_3d(image_pred, image)
            loss["psnr_3d"] += get_psnr_3d(image_pred, image)

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
            # self.writer.add_image(
            #    "eval/density (row1: gt, row2: pred)",
            #    cast_to_image(show_density),
            #    global_step,
            #    dataformats="HWC",
            # )

            proj_pred_origin_dir = osp.join(self.expdir, f"{index}_proj_pred_origin")
            proj_gt_origin_dir = osp.join(self.expdir, f"{index}_proj_gt_origin")
            proj_pred_dir = osp.join(self.expdir, f"{index}_proj_pred")
            proj_gt_dir = osp.join(self.expdir, f"{index}_proj_gt")
            # os.makedirs(eval_save_dir, exist_ok=True)
            os.makedirs(proj_pred_origin_dir, exist_ok=True)
            os.makedirs(proj_gt_origin_dir, exist_ok=True)
            os.makedirs(proj_pred_dir, exist_ok=True)
            os.makedirs(proj_gt_dir, exist_ok=True)

            # for i in tqdm(range(N)):
            #    """
            #    cast_to_image 自带了归一化, 1 - 放在外边
            #    """
            #    iio.imwrite(
            #        osp.join(proj_pred_origin_dir, f"proj_pred_{str(i)}.png"),
            #        (cast_to_image(projs_pred[i]) * 255).astype(np.uint8),
            #    )
            #    iio.imwrite(
            #        osp.join(proj_gt_origin_dir, f"proj_gt_{str(i)}.png"),
            #        (cast_to_image(projs[i]) * 255).astype(np.uint8),
            #    )
            #    iio.imwrite(
            #        osp.join(proj_pred_dir, f"proj_pred_{str(i)}.png"),
            #        ((1 - cast_to_image(projs_pred[i])) * 255).astype(np.uint8),
            #    )
            #    iio.imwrite(
            #        osp.join(proj_gt_dir, f"proj_gt_{str(i)}.png"),
            #        ((1 - cast_to_image(1 - projs[i])) * 255).astype(np.uint8),
            #    )

            ## stx()
            # for ls in loss.keys():
            #    self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)

            # Save
            # 保存各种视图
            eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
            os.makedirs(eval_save_dir, exist_ok=True)
            # np.save(
            #    osp.join(eval_save_dir, f"{index}image_pred.npy"),
            #    image_pred.cpu().detach().numpy(),
            # )
            # np.save(
            #    osp.join(eval_save_dir, f"{index}image_gt.npy"),
            #    image.cpu().detach().numpy(),
            # )
            output = np.clip(image_pred.cpu().detach().numpy(), 0, 1)
            gt = np.clip(image.cpu().detach().numpy(), 0, 1)
            output *= 255.0
            gt *= 255.0
            output = output.astype(np.uint8)
            gt = gt.astype(np.uint8)
            save_path = os.path.join(eval_save_dir, f"{index}.nii.gz")
            gt_save_path = os.path.join(eval_save_dir, f"{index}_gt.nii.gz")
            save_nifti(output, save_path)
            save_nifti(gt, gt_save_path)

            iio.imwrite(
                osp.join(eval_save_dir, f"{index}slice_show_row1_gt_row2_pred.png"),
                (cast_to_image(show_density) * 255).astype(np.uint8),
            )
            with open(osp.join(eval_save_dir, "stats.txt"), "w") as f:
                for key, value in loss.items():
                    f.write("%s: %f\n" % (key, value))

        if loss["ssim_3d"] > self.best_ssim_3d:
            torch.save(
                {
                    "epoch": idx_epoch,
                    "network": self.net.state_dict(),
                    # "network_fine": self.net_fine.state_dict() if self.n_fine > 0 else None,
                    "optimizer": self.optimizer.state_dict(),
                },
                self.ckpt_best_dir,
            )
            self.best_ssim_3d = loss["ssim_3d"]
            self.logger.info(
                f"best model update, epoch:{idx_epoch}, best 3d ssim_3d:{self.best_ssim_3d:.4g}"
            )

            # stx()

            # Logging

        return loss


trainer = BasicTrainer()
# 这并不是多线程中的start函数，而是父类Trainer中的start函数
trainer.start()  # loop train and evaluation
