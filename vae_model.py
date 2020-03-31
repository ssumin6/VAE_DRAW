import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.nn import functional as F
import numpy as np

class VAEModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.A = params['A']
        self.B = params['B']
        self.z_size = params['z_size']
        self.device = params['device']
        self.channel = params['channel']

    def encode(self, x):
        h# TODO 
        return None

    def reparameterize(self, mu, logvar):
        # TODO 
        return None

    def decode(self, z):
        # TODO 
        return None

    def forward(self, x):
        # TODO
        return None

    def loss(self, x):
        # TODO
        return None
    