import torch
import numpy as np


def calc_mse_loss(loss, x, y, k=1.0):
    """
    Calculate mse loss.
    """
    # Compute loss
    loss_mse = torch.mean((x - y) ** 2)
    loss["loss"] += k * loss_mse
    loss["loss_mse"] = loss_mse
    return loss


def calc_mse_loss_raw(loss, x, y, k=1):
    """
    Calculate mse loss for raw.
    """
    # Compute loss for raw
    loss_mse_raw = torch.mean((x - y) ** 2)
    loss["loss"] += k * loss_mse_raw
    loss["loss_mse_raw"] = loss_mse_raw
    return loss


def calc_tv_loss(loss, x, k):
    """
    Calculate total variation loss.
    Args:
        x (n1, n2, n3, 1): 3d density field.
        k: relative weight
    """
    n1, n2, n3 = x.shape
    tv_1 = torch.abs(x[1:, 1:, 1:] - x[:-1, 1:, 1:]).sum()
    tv_2 = torch.abs(x[1:, 1:, 1:] - x[1:, :-1, 1:]).sum()
    tv_3 = torch.abs(x[1:, 1:, 1:] - x[1:, 1:, :-1]).sum()
    tv = (tv_1 + tv_2 + tv_3) / (n1 * n2 * n3)
    loss["loss"] += tv * k
    loss["loss_tv"] = tv * k
    return loss


def compute_tv_norm(
    values, losstype="l2", weighting=None
):  # pylint: disable=g-doc-args
    """Returns TV norm for input values.

    Note: The weighting / masking term was necessary to avoid degenerate
    solutions on GPU; only observed on individual DTU scenes.
    """
    v00 = values[:, :-1, :-1]
    v01 = values[:, :-1, 1:]
    v10 = values[:, 1:, :-1]

    if losstype == "l2":
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    elif losstype == "l1":
        loss = np.abs(v00 - v01) + np.abs(v00 - v10)
    else:
        raise ValueError("Not supported losstype.")

    if weighting is not None:
        loss = loss * weighting
    return loss
