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

        self.fc1 = nn.Linear(self.A*self.A, 400)
        self.fc21 = nn.Linear(400, self.z_size)
        self.fc22 = nn.Linear(400, self.z_size)
        self.fc3 = nn.Linear(self.z_size, 400)
        self.fc4 = nn.Linear(400, self.A*self.A)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x):
        

        x_recon, mu, logvar= self.forward(x)
        Lx = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        Ld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return Lx + Ld 
    