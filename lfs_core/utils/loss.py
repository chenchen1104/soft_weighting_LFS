import torch
import torch.nn.functional as F
import math
from models.losses import *


def loss_search(outputs, targets, inputs, a):
    cd = get_loss_layer('cd')
    emd = get_loss_layer('emd')
    Repulsion = get_loss_layer('repu')
    # 必须写加法！！！
    loss = a[0].item() * cd(outputs, targets) + a[1].item() * emd(outputs, targets) + a[2] * Repulsion(outputs)
    return loss.sum()


def loss_baseline(outputs, targets):
    cd = get_loss_layer('cd')
    emd = get_loss_layer('emd')
    loss = cd(outputs, targets) + emd(outputs, targets)
    return loss.sum()
