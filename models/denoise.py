import sys
import torch
from torch import nn
import numpy as np

from .net import *
from .losses import *


class PointCloudDenoising(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        try:
            random_mesh = True if hparams.random_mesh else False
        except AttributeError:
            random_mesh = False

        try:
            random_pool = True if hparams.random_pool else False
        except AttributeError:
            random_pool = False

        try:
            random_pool = True if hparams.no_prefilter else False
        except AttributeError:
            no_prefilter = False

        self.model = DenoiseNet(
            loss_rec=hparams.loss_rec,
            loss_ds=hparams.loss_ds if hparams.loss_ds != 'None' else None,
            activation=hparams.activation,
            conv_knns=[int(k) for k in hparams.knn.split(',')],
            gpool_use_mlp=True if hparams.gpool_mlp else False,
            dynamic_graph=False if hparams.static_graph else True,
            use_random_mesh=random_mesh,
            use_random_pool=random_pool
        )

        # For validation
        self.cd_loss = ChamferLoss()
        self.emd_loss = EMDLoss(eps=0.005, iters=50)
        self.P2S = P2SLoss()

    def forward(self, pos):
        return self.model(pos)

    def getloss(self, denoised, noiseless, input):
        loss = self.model.get_loss(gts=noiseless, preds=denoised, inputs=input)
        return loss
