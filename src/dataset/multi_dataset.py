import torch
import pickle
import os
import sys
import numpy as np
from functools import cache
import json
import yaml

from torch.utils.data import DataLoader, Dataset
from pdb import set_trace as stx
from src.render import get_pts
import SimpleITK as sitk
from copy import deepcopy


def read_nifti(path):
    itk_img = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(itk_img)
    return image


def save_nifti(image, path):
    out = sitk.GetImageFromArray(image)
    sitk.WriteImage(out, path)


def load_names():
    with open("info.json", "r") as f:
        info = json.load(f)
        names = []
        for s in ["train", "test", "eval"]:
            names += info[s]
        return names


def coord_to_dif(points):
    return ((points + 0.1275) / (0.1275 + 0.1275) * 2) - 1


def coord_to_dif_base(points):
    return (points + 0.1275) / (0.1275 + 0.1275)


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


# 这里的各项参数代表的物理含义可以在哪查到呢？
class ConeGeometry(object):
    """
    Cone beam CT geometry. Note that we convert to meter from millimeter. 1 m = 1000 mm
    """

    def __init__(self, data):

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = (
            data["DSD"] / 1000
        )  # Distance Source to Detector      (m) x射线发射源到x射线接收器之间的距离
        self.DSO = (
            data["DSO"] / 1000
        )  # Distance Source Origin        (m) 发射源到起点之间的距离

        # Detector parameters

        self.nDetector = np.array(
            data["nDetector"]
        )  # number of pixels              (px)
        self.dDetector = (
            np.array(data["dDetector"]) / 1000
        )  # size of each pixel            (m)
        self.sDetector = (
            self.nDetector * self.dDetector
        )  # total size of the detector    (m)

        # Image parameters
        self.nVoxel = np.array(data["nVoxel"])  # number of voxels              (vx)
        self.dVoxel = (
            np.array(data["dVoxel"]) / 1000
        )  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = (
            np.array(data["offOrigin"]) / 1000
        )  # Offset of image from origin   (m)
        self.offDetector = (
            np.array(data["offDetector"]) / 1000
        )  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data[
            "accuracy"
        ]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]

        self.v_res = data["nVoxel"][0]  # ct scan
        self.p_res = data["nDetector"][0]  # projections
        self.v_spacing = np.array(data["dVoxel"])[0]  # mm
        self.p_spacing = np.array(data["dDetector"])[0]  # mm

    def project(self, points, angle):
        # points: [N, 3] ranging from [0, 1]
        # d_points: [N, 2] ranging from [-1, 1]

        points = deepcopy(points)
        points = points.cpu().detach().numpy()
        points[:, :2] -= 0.5  # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2]  # [-0.5, 0.5]
        points *= self.v_res * self.v_spacing  # mm

        angle = -1 * angle  # inverse direction
        rot_M = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

        points = points @ rot_M.T

        d1 = self.DSO * 1000
        d2 = self.DSD * 1000

        coeff = (d2) / (d1 - points[:, 0])  # N,
        d_points = points[:, [2, 1]] * coeff[:, None]  # [N, 2] float
        d_points /= self.p_res * self.p_spacing
        d_points *= 2  # NOTE: some points may fall outside [-1, 1]

        return d_points


# dataloader，把数据做成 TIGRE 数据类型
class MultiTIGREDataset(Dataset):
    """
    TIGRE dataset.
    """

    def __init__(
        self, data_dir, n_rays=1024, n_samples=512, type="train", device="cuda"
    ):
        super().__init__()
        self.data_dir = data_dir

        with open(os.path.join(data_dir, "info.json"), "r") as f:
            cfg = json.load(f)
        self.cfg = cfg

        # stx()
        with open(self.cfg["config.yaml"], "rb") as handle:
            data = yaml.load(handle, Loader=yaml.FullLoader)
        self.geo = ConeGeometry(data)  # 把数据处理成ConeGeometry
        self.type = type
        self.n_rays = n_rays
        self.near, self.far = self.get_near_far(self.geo)
        self.n_views = self.cfg["n_views"]
        self.n_samples = n_samples
        self.device = device
        self.voxels = torch.tensor(
            self.get_voxels(self.geo), dtype=torch.float32, device=device
        )
        self.angles = np.linspace(0, 180 / 180 * np.pi, self.n_views + 1)[:-1]
        rays = self.get_rays(
            self.angles, self.geo, device
        )  # [50, 256, 256, 6] 在每一个角度下获取射线的原点和方向
        self.rays = torch.cat(
            [
                rays,
                torch.ones_like(rays[..., :1]) * self.near,
                torch.ones_like(rays[..., :1]) * self.far,
            ],
            dim=-1,
        )

        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(
                    0,
                    self.geo.nDetector[1] - 1,
                    self.geo.nDetector[1],
                    device=device,
                ),
                torch.linspace(
                    0,
                    self.geo.nDetector[0] - 1,
                    self.geo.nDetector[0],
                    device=device,
                ),
                indexing="ij",
            ),
            -1,
        )
        self.coords = torch.reshape(coords, [-1, 2])

    def __len__(self):
        if self.type == "train":
            return len(self.cfg["train"])
        else:
            return len(self.cfg["eval"])

    def __getitem__(self, index):
        if self.type == "train":
            # stx()
            """
            d['projs'] - [10, 256, 256]
            """
            name = self.cfg["train"][index]
            image_path = self.cfg["image"].format(name)
            image = read_nifti(image_path)
            image = torch.tensor(image, dtype=torch.float32, device=self.device)
            projection_path = self.cfg["projections"].format(name)
            projections = pickle.load(open(projection_path, "rb"))
            projections = torch.tensor(
                projections, dtype=torch.float32, device=self.device
            )
            projs_list = []
            rays_list = []
            pts_list = []
            image_pts_list = []
            coords_list = []
            for proj_num in range(self.n_views):
                projs_valid = (projections[proj_num] > 0).flatten()
                coords_valid = self.coords[
                    projs_valid
                ]  # [65536, 2] -> [40653, 2], 将布尔值矩阵当做索引，可能是因为并不是所有的
                select_inds = np.random.choice(
                    coords_valid.shape[0], size=[self.n_rays], replace=False
                )  # 从 0 ~ 40653-1 中选取 1024 个值
                select_coords = coords_valid[
                    select_inds
                ].long()  # 根据选取的索引值来取坐标
                rays = self.rays[
                    proj_num, select_coords[:, 0], select_coords[:, 1]
                ]  # self.rays: [50, 256, 256, 6], index 决定了取哪一个角度或样例，后两项决定了横纵坐标
                projs = projections[proj_num, select_coords[:, 0], select_coords[:, 1]]
                pts, _, _, _ = get_pts(
                    rays,
                    10,
                )
                pts = pts.reshape(-1, 3)
                q = coord_to_dif_base(pts)
                cl = []
                for other_proj_num in range(self.n_views):
                    coords = self.geo.project(q, self.angles[other_proj_num])
                    coords = torch.tensor(
                        coords, dtype=torch.float32, device=self.device
                    )
                    cl.append(coords)
                coords = torch.stack(cl, dim=0)
                #
                image_pts = index_3d(image, pts)
                projs_list.append(projs)
                rays_list.append(rays)
                pts_list.append(pts)
                image_pts_list.append(image_pts)
                coords_list.append(coords)

            image_pts = torch.stack(image_pts_list, dim=0)
            pts = torch.stack(pts_list, dim=0)
            rays = torch.stack(rays_list, dim=0)
            projs = torch.stack(projs_list, dim=0)
            coords = torch.stack(coords_list, dim=0)
            coords = coords.permute(1, 0, 2, 3)
            coords = coords.reshape(self.n_views, -1, 2)

            return {
                "projs": projs,
                "rays": rays,
                "pts:": pts,
                "image": image_pts,
                "projections": projections,
                "proj_pts": coords,
            }

        elif self.type == "val":
            name = self.cfg["eval"][index]
            image_path = self.cfg["image"].format(name)
            image = read_nifti(image_path)
            image = torch.tensor(image, dtype=torch.float32, device=self.device)
            projection_path = self.cfg["projections"].format(name)
            projections = pickle.load(open(projection_path, "rb"))
            projections = torch.tensor(
                projections, dtype=torch.float32, device=self.device
            )
            projs_list = []
            rays_list = []
            pts_list = []
            image_pts_list = []
            coords_list = []
            for proj_num in range(self.n_views):
                projs_valid = (projections[proj_num] > 0).flatten()
                coords_valid = self.coords[
                    projs_valid
                ]  # [65536, 2] -> [40653, 2], 将布尔值矩阵当做索引，可能是因为并不是所有的
                rays = self.rays[
                    proj_num
                ]  # self.rays: [50, 256, 256, 6], index 决定了取哪一个角度或样例，后两项决定了横纵坐标
                projs = projections[proj_num]
                pts, _, _, _ = get_pts(
                    rays,
                    self.n_samples,
                )
                pts = pts.reshape(-1, 3)
                image_pts = index_3d(image, pts)
                projs_list.append(projs)
                rays_list.append(rays)
                pts_list.append(pts)
                image_pts_list.append(image_pts)
                coords_list.append(select_coords)
            image_pts = torch.stack(image_pts_list, dim=0)
            pts = torch.stack(pts_list, dim=0)
            rays = torch.stack(rays_list, dim=0)
            projs = torch.stack(projs_list, dim=0)
            coords = torch.stack(coords_list, dim=0)
            return {
                "projs": projs,
                "rays": rays,
                "pts:": pts,
                "image": image_pts,
                "projections": projections,
                "proj_pts": coords,
            }
        print("??????")
        return {}

    # 此处的 geo: ConeGeometry 表示什么？圆锥形几何
    # 冒号是类型建议符，告诉程序员希望传入的实参的类型
    @cache
    def get_voxels(self, geo: ConeGeometry):
        """
        Get the voxels.
        """
        n1, n2, n3 = geo.nVoxel
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2  # 这个参数是什么意思？

        """
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"])  # number of voxels              (vx)
        self.dVoxel = np.array(data["dVoxel"])/1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)
        """
        # xyz 是一个多层嵌套list，外部是3, 256, 256, 178
        xyz = np.meshgrid(
            np.linspace(-s1, s1, n1),
            np.linspace(-s2, s2, n2),
            np.linspace(-s3, s3, n3),
            indexing="ij",
        )
        # voxel 就是一个 numpy.ndarray, 形状为 (256, 256, 178, 3)
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        # stx()
        return voxel

    # 从哪可以看
    # H, W 信息融合在了 ConeGeometry 内部
    """
        一般而言, 由一张RGB图片找到渲染其的一簇rays的流程是
        (1) 像素坐标系转相机坐标系
        (2) 相机坐标系转世界坐标系，分别确定射线的源点(rays_o)和方向(rays_d)
    """

    def get_rays(self, angles, geo: ConeGeometry, device):
        """
        Get rays given one angle and x-ray machine geometry.
        """

        W, H = geo.nDetector
        DSD = geo.DSD
        rays = []

        for angle in angles:
            pose = torch.Tensor(self.angle2pose(geo.DSO, angle)).to(device)
            rays_o, rays_d = None, None
            if geo.mode == "cone":
                # 构造像素坐标系, 创建grid
                i, j = torch.meshgrid(
                    torch.linspace(0, W - 1, W, device=device),
                    torch.linspace(0, H - 1, H, device=device),
                    indexing="ij",
                )  # pytorch"s meshgrid has indexing="ij"
                # 以中心点为圆心的像素坐标系
                """
                    geo.dDetector    ——  size of each pixel
                    geo.offDetector  ——  Offset of Detector from origin, 一般等于 DSD - DSO
                    一般会以 origin 作为圆心建立xy轴, 然后垂直纸面朝外为z轴
                """
                # 把像素坐标转换成对origin的横纵偏移
                """
                    如果xy和uv对齐, 那么下面这两条式子就可以说通了
                    看着像是像素坐标系转成相机坐标系
                """
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]

                # 为啥要除以DSD呢？不清楚，xy方向确定了，z方向为何stack个1呢？
                # 除以normalize是为了normalize吗？
                dirs = torch.stack(
                    [uu / DSD, vv / DSD, torch.ones_like(uu)], -1
                )  # 由像素上的一些坐标确定射线簇的方向

                # source to origin 或者 origin to source
                rays_d = torch.sum(
                    torch.matmul(pose[:3, :3], dirs[..., None]).to(device), -1
                )  # pose[:3, :3] *
                rays_o = pose[:3, -1].expand(rays_d.shape)  # 相机偏移量为射线的源点

            elif geo.mode == "parallel":
                i, j = torch.meshgrid(
                    torch.linspace(0, W - 1, W, device=device),
                    torch.linspace(0, H - 1, H, device=device),
                    indexing="ij",
                )  # pytorch"s meshgrid has indexing="ij"
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack(
                    [torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)],
                    -1,
                )  # 与cone geometry的区别在于
                rays_d = torch.sum(
                    torch.matmul(pose[:3, :3], dirs[..., None]).to(device), -1
                )  # pose[:3, :3] *
                rays_o = torch.sum(
                    torch.matmul(
                        pose[:3, :3],
                        torch.stack([uu, vv, torch.zeros_like(uu)], -1)[..., None],
                    ).to(device),
                    -1,
                ) + pose[:3, -1].expand(rays_d.shape)

                # import open3d as o3d
                # from src.util.draw_util import plot_rays, plot_cube, plot_camera_pose
                # cube1 = plot_cube(np.zeros((3,1)), geo.sVoxel[...,np.newaxis])
                # cube2 = plot_cube(np.zeros((3,1)), np.ones((3,1))*geo.DSO*2)
                # rays1 = plot_rays(rays_d.cpu().detach().numpy(), rays_o.cpu().detach().numpy(), 2)
                # poseray = plot_camera_pose(pose.cpu().detach().numpy())
                # o3d.visualization.draw_geometries([cube1, cube2, rays1, poseray])

            else:
                raise NotImplementedError("Unknown CT scanner type!")
            rays.append(torch.concat([rays_o, rays_d], dim=-1))

        return torch.stack(rays, dim=0)

    # world to camera transer ？
    # 相机外参矩阵
    # 绕x轴旋转90度，绕z轴旋转90度后再旋转angle
    # 前两次旋转为对齐
    # 此处为 source_to_origin 的pose
    def angle2pose(self, DSO, angle):

        # 绕x轴逆时针转了-90度
        phi1 = -np.pi / 2
        R1 = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(phi1), -np.sin(phi1)],
                [0.0, np.sin(phi1), np.cos(phi1)],
            ]
        )

        # 绕z轴，逆时针转了90度
        phi2 = np.pi / 2
        R2 = np.array(
            [
                [np.cos(phi2), -np.sin(phi2), 0.0],
                [np.sin(phi2), np.cos(phi2), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        # 绕z轴，逆时针转了angle
        R3 = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rot = np.dot(np.dot(R3, R2), R1)

        # source 对应的偏移而非相机 (detector) 对应的偏移
        # 不沿着z轴运动，所以z轴偏移量为0
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])  # DSO 投影
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans  # 偏移量参数
        return T

    # 根据 CT 的锥形几何计算近端和远端
    # near 和 far 是由什么决定的呢？很奇怪
    # tolerance是允许的误差范围
    """
        Numpy中的 linalg 模块包含线性代数中的函数方法，用于求解矩阵的逆矩阵、求特征值、解线性方程组以及求行列式等
    """

    def get_near_far(self, geo: ConeGeometry, tolerance=0.005):
        """
        Compute the near and far threshold.
        self.offOrigin = np.array(data["offOrigin"])/1000       #   Offset of image from origin
        self.sVoxel = self.nVoxel * self.dVoxel                 #
        """
        dist1 = np.linalg.norm(
            [geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2]
        )
        dist2 = np.linalg.norm(
            [geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2]
        )
        dist3 = np.linalg.norm(
            [geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2]
        )
        dist4 = np.linalg.norm(
            [geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2]
        )
        dist_max = np.max([dist1, dist2, dist3, dist4])
        near = np.max([0, geo.DSO - dist_max - tolerance])
        far = np.min([geo.DSO * 2, geo.DSO + dist_max + tolerance])
        return near, far
