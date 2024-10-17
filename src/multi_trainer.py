import os
import os.path as osp
import json
import torch
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm, trange
from shutil import copyfile
import numpy as np
from torch.utils.data import DataLoader

from .dataset import MultiTIGREDataset as Dataset

from .utils import gen_log, time2file_name
import datetime

from .network import get_network
from .encoder import get_encoder
from pdb import set_trace as stx


class Trainer:
    def __init__(self, cfg, device="cuda"):

        # Args，从配置文件中导入各项参数
        self.global_step = 0
        self.conf = cfg
        self.epochs = cfg["train"]["epoch"]
        self.i_eval = cfg["log"]["i_eval"]          # epoch for evaluation
        self.i_save = cfg["log"]["i_save"]          # epoch for saving
        self.netchunk = cfg["render"]["netchunk"]   
        self.n_rays = cfg["train"]["n_rays"]
        
  
        # Log direcotry，设置实验路径和文件夹
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
        self.expdir = osp.join(cfg["exp"]["expdir"], cfg["exp"]["expname"], date_time)
        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.ckpt_best_dir = osp.join(self.expdir, "ckpt_best.tar")
        self.best_psnr_3d = 0
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.evaldir, exist_ok=True)
        self.logger = gen_log(self.expdir)
        self.net_fine = None

        # Dataset，读数据，dataloader
        '''
            eval 和 train dataset 并不相同
        '''
        train_dataset = Dataset(cfg["exp"]["train_datadir"], cfg["train"]["n_rays"], "train", device) # 由dataset去构造数据集
        val_dataset = Dataset(cfg["exp"]["eval_datadir"], cfg["train"]["n_rays"], "val", device) # 由val
        # stx()
        self.train_dloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        self.eval_dset = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

        #train_dataset, val_dataset = torch.utils.data.random_split(dataset, [110, 21])
    
        # Network，实例化网络
        network = get_network(cfg["network"]["net_type"])
        cfg["network"].pop("net_type", None)
        encoder = get_encoder(**cfg["encoder"])
        # stx()
        self.net = network(encoder, **cfg["network"]).to(device)
        grad_vars = list(self.net.parameters())
        
        # Optimizer，优化器及LR策略
        # optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
        '''
            optimizer 更新权重 weights, 用的是 optimizer.step()
            scheduler 更新学习率 lr, 用的是 scheduler.step()
        '''
        self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg["train"]["lrate"], betas=(0.9, 0.999))
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer, gamma=cfg["train"]["lrate_gamma"])
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=cfg["train"]["lrate_step"], gamma=cfg["train"]["lrate_gamma"])

        # Load checkpoints
        self.epoch_start = 0
        if cfg["train"]["resume"] and osp.exists(self.ckptdir):
            print(f"Load checkpoints from {self.ckptdir}.")
            ckpt = torch.load(self.ckptdir)
            self.epoch_start = ckpt["epoch"] + 1
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.global_step = self.epoch_start * len(self.train_dloader)
            self.net.load_state_dict(ckpt["network"])

        # Summary writer 需要用tensorboard打开来看，不如直接就txt文件记录
        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)

    def args2string(self, hp):
        """
        Transfer args to string.
        """
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    def start(self):
        """
        Main loop.
        """
        self.logger.info(self.conf)
        def fmt_loss_str(losses):
            return "".join(", " + k + ": " + f"{losses[k].item():.4g}" for k in losses)
        iter_per_epoch = len(self.train_dloader)
        pbar = tqdm(total= iter_per_epoch * self.epochs, leave=True)    # processing bar
        if self.epoch_start > 0:
            pbar.update(self.epoch_start*iter_per_epoch)        # 更新进度条

        for idx_epoch in range(self.epoch_start, self.epochs+1):

            # Evaluate
            if (idx_epoch % self.i_eval == 0 or idx_epoch == self.epochs) and self.i_eval > 0 and idx_epoch > 0:
                self.net.eval()             # self.net 和 self.net_fine 分别表示粗细网络
                with torch.no_grad():
                    loss_test = self.eval_step(global_step=self.global_step, idx_epoch=idx_epoch)
                self.net.train()
                tqdm.write(f"[EVAL] epoch: {idx_epoch}/{self.epochs}{fmt_loss_str(loss_test)}")  # 此处为何不报PSNR？
                self.logger.info(f"[EVAL] epoch: {idx_epoch}/{self.epochs}{fmt_loss_str(loss_test)}")
            
            # Train
            # stx()
            for data in self.train_dloader:
                self.global_step += 1
                # Train
                self.net.train()
                loss_train = self.train_step(data, global_step=self.global_step, idx_epoch=idx_epoch)
                pbar.set_description(f"epoch={idx_epoch}/{self.epochs}, loss={loss_train:.4g}, lr={self.optimizer.param_groups[0]['lr']:.4g}")
                pbar.update(1)
            
            if idx_epoch % 10 == 0:
                self.logger.info(f"epoch={idx_epoch}/{self.epochs}, loss={loss_train:.4g}, lr={self.optimizer.param_groups[0]['lr']:.4g}")
            
            # Save
            if (idx_epoch % self.i_save == 0 or idx_epoch == self.epochs) and self.i_save > 0 and idx_epoch > 0:
                if osp.exists(self.ckptdir):
                    copyfile(self.ckptdir, self.ckptdir_backup)
                tqdm.write(f"[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}")
                self.logger.info(f"[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}")
                torch.save(
                    {
                        "epoch": idx_epoch,
                        "network": self.net.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self.ckptdir,
                ) # 此处并没有save best的操作呀

            # Update lrate
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            # self.logger.info(f"train/lr: {self.optimizer.param_groups[0]["lr"]},{self.global_step}")
            self.lr_scheduler.step()

        tqdm.write(f"Training complete! See logs in {self.expdir}")

    def train_step(self, data, global_step, idx_epoch):
        """
        Training step
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, global_step, idx_epoch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    # 下面两个函数在父类中不作定义，在子类中进行重写

    def compute_loss(self, data, global_step, idx_epoch):
        """
        Training step
        """
        raise NotImplementedError()


    def eval_step(self, global_step, idx_epoch):
        """
        Evaluation step
        """
        raise NotImplementedError()
        
