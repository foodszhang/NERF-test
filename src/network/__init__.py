from .network import DensityNetwork
from .Lineformer import Lineformer
from .mix import DIF_Net, ImageNerfNetwork


def get_network(type):
    if type == "mlp":
        return DensityNetwork
    elif type == "Lineformer":
        return Lineformer
    elif type == "dif":
        return DIF_Net
    elif type == "ImageNerf":
        return ImageNerfNetwork
    else:
        raise NotImplementedError("Unknown network type!")
