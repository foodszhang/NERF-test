import os
import os.path as osp
import tigre
from tigre.utilities.geometry import Geometry
from tigre.utilities import gpu
import numpy as np
import yaml
import SimpleITK as sitk

import pickle
import scipy.io
import scipy.ndimage.interpolation
from tigre.utilities import CTnoise

import skimage as ski
import cv2

import argparse
import imageio.v2 as iio
import json


def read_nifti(path):
    itk_img = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(itk_img)
    return image


def save_nifti(image, path):
    out = sitk.GetImageFromArray(image)
    sitk.WriteImage(out, path)


def main():
    # matPath = f"./dataGenerator/{dataFolder}/{dataType}/img.mat"
    data_dir = "./data/"
    configPath = f"./config.yml"
    infoPath = osp.join(data_dir, "info.json")
    info = json.load(open(infoPath, "r"))
    all_names = info["train"] + info["eval"] + info["test"]
    for name in all_names:
        generator(name, data_dir, configPath, "luna16", show=False)


# %% Geometry
class ConeGeometry_special(Geometry):
    """
    Cone beam CT geometry.
    """

    def __init__(self, data):
        Geometry.__init__(self)

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"] / 1000  # Distance Source Detector      (m)
        self.DSO = data["DSO"] / 1000  # Distance Source Origin        (m)
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
        self.nVoxel = np.array(
            data["nVoxel"][::-1]
        )  # number of voxels              (vx)
        self.dVoxel = (
            np.array(data["dVoxel"][::-1]) / 1000
        )  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = (
            np.array(data["offOrigin"][::-1]) / 1000
        )  # Offset of image from origin   (m)
        self.offDetector = (
            np.array([data["offDetector"][1], data["offDetector"][0], 0]) / 1000
        )  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data[
            "accuracy"
        ]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]


"""
    将 HU 装换成 attenuation
"""


def convert_to_attenuation(
    data: np.array, rescale_slope: float, rescale_intercept: float
):
    """
    CT scan is measured using Hounsfield units (HU). We need to convert it to attenuation.

    The HU is first computed with rescaling parameters:
        HU = slope * data + intercept

    Then HU is converted to attenuation:
        mu = mu_water + HU/1000x(mu_water-mu_air)
        mu_water = 0.206
        mu_air=0.0004

    Args:
    data (np.array(X, Y, Z)): CT data.
    rescale_slope (float): rescale slope.
    rescale_intercept (float): rescale intercept.

    Returns:
    mu (np.array(X, Y, Z)): attenuation map.

    """
    HU = data * rescale_slope + rescale_intercept
    mu_water = 0.206
    mu_air = 0.0004
    mu = mu_water + (mu_water - mu_air) / 1000 * HU
    # mu = mu * 100
    return mu


def loadImage(
    dirname,
    nVoxels,
):
    """
    Load CT image.
    """

    if nVoxels is None:
        nVoxels = np.array((256, 256, 256))

    # test_data = scipy.io.loadmat(dirname)       # 加载 img.mat 文件
    itk_image = sitk.ReadImage(dirname)
    test_data = sitk.GetArrayFromImage(itk_image)
    spacing = itk_image.GetSpacing()
    spacing = (spacing[2], spacing[1], spacing[0])

    # Loads data in F_CONTIGUOUS MODE (column major), convert to Row major
    image = test_data.astype(np.float32)

    imageDim = image.shape

    zoom_x = nVoxels[0] / imageDim[0]
    zoom_y = nVoxels[1] / imageDim[1]
    zoom_z = nVoxels[2] / imageDim[2]

    """
        根据体素个数与图像维度的比值来进行缩放
    """
    if zoom_x != 1.0 or zoom_y != 1.0 or zoom_z != 1.0:
        print(
            f"Resize ct image from {imageDim[0]}x{imageDim[1]}x{imageDim[2]} to "
            f"{nVoxels[0]}x{nVoxels[1]}x{nVoxels[2]}"
        )
        image = scipy.ndimage.interpolation.zoom(
            image, (zoom_x, zoom_y, zoom_z), order=3, prefilter=False
        )

    return image


def generator(name, data_dir, configPath, result_dir, show=False):
    """
    Generate projections given CT image and configuration.

    """

    # Load configuration
    with open(configPath, "r") as handle:
        data = yaml.safe_load(handle)

    # Load CT image
    matPath = osp.join(data_dir, name)
    geo = ConeGeometry_special(data)
    img = loadImage(
        matPath,
        data["nVoxel"],
    )
    window = (-800, 1000)
    img = convert_to_attenuation(img, data["rescale_slope"], data["rescale_intercept"])
    window = (
        convert_to_attenuation(
            window[0], data["rescale_slope"], data["rescale_intercept"]
        ),
        convert_to_attenuation(
            window[1], data["rescale_slope"], data["rescale_intercept"]
        ),
    )
    img = np.clip(img, window[0], window[1])
    img = (img - window[0]) / (window[1] - window[0])
    image_result_dir = osp.join(data_dir, result_dir, "image")
    projection_result_dir = osp.join(data_dir, result_dir, "projection")
    os.makedirs(image_result_dir, exist_ok=True)
    os.makedirs(projection_result_dir, exist_ok=True)
    img = np.transpose(img, (2, 1, 0))
    save_nifti(img, osp.join(image_result_dir, f"{name}.nii.gz"))

    data["angles"] = (
        np.linspace(0, data["totalAngle"] / 180 * np.pi, data["numTrain"] + 1)[:-1]
        + data["startAngle"] / 180 * np.pi
    )
    projections = tigre.Ax(np.transpose(img, (2, 1, 0)).copy(), geo, data["angles"])[
        :, ::-1, :
    ]
    with open(osp.join(projection_result_dir, f"{name}.pickle"), "wb") as handle:
        pickle.dump(projections, handle, pickle.HIGHEST_PROTOCOL)
    if show or True:
        save_dir_train_ct = osp.join("dataGenerator/", result_dir, "show_vis_train_ct/")
        save_dir_train_proj = osp.join(
            "dataGenerator/", result_dir, "show_vis_train_proj/"
        )
        save_dir_vali_proj = osp.join(
            "dataGenerator/", result_dir, "show_vis_vali_proj/"
        )

        os.makedirs(save_dir_train_ct, exist_ok=True)
        os.makedirs(save_dir_train_proj, exist_ok=True)
        os.makedirs(save_dir_vali_proj, exist_ok=True)
        # stx()
        """
            img: [256, 256, 128]
            data["train"]["projections"]: 50, 512, 512
            data["val"]["projections"]: 50, 512, 512
        """

        show_step = 1
        show_num = projections.shape[0] // show_step
        show_image_train_ct = img[..., ::show_step]
        show_dir_train_proj = projections[::show_step, ...]
        # show_image = np.concatenate(show_image, axis=0)

        # stx()

        for i in range(show_num):
            iio.imwrite(
                save_dir_train_ct + f"CT_{name}_" + str(i) + ".png",
                (show_image_train_ct[..., i] * 255 / show_image_train_ct.max()).astype(
                    np.uint8
                ),
            )
            iio.imwrite(
                save_dir_train_proj + f"projs_{name}_" + str(i) + ".png",
                (show_dir_train_proj[i, ...] * 255 / show_dir_train_proj.max()).astype(
                    np.uint8
                ),
            )
    return data


def multi_gen(dir_path, configPath, dataFolder, outputFolder, dataType, show=False):
    index = 0
    for file in os.listdir(dir_path):
        if file.endswith("mhd"):
            matPath = os.path.join(dir_path, file)
            data = generator(matPath, configPath, dataFolder, dataType, index, show)
            index += 1
            outputDir = osp.join(outputFolder, f"{dataType}_{index}")
            os.makedirs(outputDir, exist_ok=True)
            outputPath = osp.join(outputDir, f"data.pickle")
            with open(outputPath, "wb") as handle:
                pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
