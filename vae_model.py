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

        # fc layers
        self.fc_enc = nn.Linear(self.A*self.B, 400)
        self.fc_enc_mu = nn.Linear(400, self.z_size)
        self.fc_enc_var = nn.Linear(400, self.z_size)

        self.fc_dec = nn.Linear(self.z_size, 400)
        self.fc_dec2 = nn.Linear(400, self.A*self.B)

    def encode(self, x):
        h = F.relu(self.fc_enc(x))
        mu = self.fc_enc_mu(h)
        logvar = self.fc_enc_var(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        x = torch.randn_like(std)
        return mu + x*std

    def decode(self, z):
        h = F.relu(self.fc_dec(z))
        return torch.sigmoid(self.fc_dec2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        # calculate KL divergence for Q(z|X) and P(z)
        kl = mu.pow(2) + logvar.exp().pow(2) - 1 - logvar
        kl = torch.sum(kl).mul(0.5) 
        z = self.reparameterize(mu, logvar)
        return self.decode(z), kl

    def loss(self, x):
        # x_r : reconstructed 
        x_r, kl = self.forward(x)
        loss = F.binary_cross_entropy(x_r, x, reduction ='sum') + kl
        return loss
    

    def generate(self, num_output, epoch):
        # generates image
        img = self.decode(torch.randn(num_output, self.z_size))

        img = img.view(-1, self.channel, self.A, self.B)
        vutils.save_image(img, 'generate/sample_'+str(epoch)+'.png')