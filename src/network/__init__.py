from .network import DensityNetwork
from .Lineformer import Lineformer
from .mix import Mix_Net




def get_network(type):
    if type == "mlp":
        return DensityNetwork
    elif type == "Lineformer":
        return Lineformer
    elif type == 'mix':
        return Mix_Net
    else:
        raise NotImplementedError("Unknown network type!")

