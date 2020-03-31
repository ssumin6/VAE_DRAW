import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np

"""
The equation numbers on the comments corresponding
to the relevant equation given in the paper:
DRAW: A Recurrent Neural Network For Image Generation.
"""

class DRAWModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.T = params['T']
        self.A = params['A']
        self.B = params['B']
        self.z_size = params['z_size']
        self.read_N = params['read_N']
        self.write_N = params['write_N']
        self.enc_size = params['enc_size']
        self.dec_size = params['dec_size']
        self.device = params['device']
        self.channel = params['channel']

        # Stores the generated image for each time step.
        self.cs = [0] * self.T
        
        # To store appropriate values used for calculating the latent loss (KL-Divergence loss)
        self.logsigmas = [0] * self.T
        self.sigmas = [0] * self.T
        self.mus = [0] * self.T

        self.encoder = nn.LSTMCell(2*self.read_N*self.read_N*self.channel + self.dec_size, self.enc_size)

        # To get the mean and standard deviation for the distribution of z.
        self.fc_mu = nn.Linear(self.enc_size, self.z_size)
        self.fc_sigma = nn.Linear(self.enc_size, self.z_size)

        self.decoder = nn.LSTMCell(self.z_size, self.dec_size)

        self.fc_write = nn.Linear(self.dec_size, self.write_N*self.write_N*self.channel)

        # To get the attention parameters. 5 in total.
        self.fc_attention = nn.Linear(self.dec_size, 5)

    def forward(self, x):
        self.batch_size = x.size(0)

        # requires_grad should be set True to allow backpropagation of the gradients for training.
        h_enc_prev = torch.zeros(self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        h_dec_prev = torch.zeros(self.batch_size, self.dec_size, requires_grad=True, device=self.device)

        enc_state = torch.zeros(self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        dec_state = torch.zeros(self.batch_size, self.dec_size, requires_grad=True, device=self.device)

        for t in range(self.T):
            c_prev = torch.zeros(self.batch_size, self.B*self.A*self.channel, requires_grad=True, device=self.device) if t == 0 else self.cs[t-1]
            # Equation 3.TODO
            x_hat = None
            # Equation 4.TODO
            # Get the N x N glimpse.
            r_t = None
            # Equation 5.TODO
            h_enc, enc_state = None
            # Equation 6.
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(h_enc)
            # Equation 7.TODO
            h_dec, dec_state = None
            # Equation 8. TODO
            self.cs[t] = None

            h_enc_prev = h_enc
            h_dec_prev = h_dec

    def read(self, x, x_hat, h_dec_prev):
        # Using attention 
        (Fx, Fy), gamma = self.attn_window(h_dec_prev, self.read_N)

        def filter_img(img, Fx, Fy, gamma):
            Fxt = Fx.transpose(self.channel, 2)
            if self.channel == 3:
                img = img.view(-1, 3, self.B, self.A)
            elif self.channel == 1:
                img = img.view(-1, self.B, self.A)

            # Equation 27. TODO 
            glimpse = None

            return glimpse * gamma.view(-1, 1).expand_as(glimpse)

        x = filter_img(x, Fx, Fy, gamma)
        x_hat = filter_img(x_hat, Fx, Fy, gamma)

        return torch.cat((x, x_hat), dim=1)
        # No attention
        #return torch.cat((x, x_hat), dim=1)

    def write(self, h_dec):
        # Using attention
        # Equation 28. TODO
        w = None
        if self.channel == 3:
            w = w.view(self.batch_size, 3, self.write_N, self.write_N)
        elif self.channel == 1:
            w = w.view(self.batch_size, self.write_N, self.write_N)

        (Fx, Fy), gamma = None
        Fyt = Fy.transpose(self.channel, 2)

        # Equation 29. TODO
        wr = None

        return wr 
        # No attention
        #return self.fc_write(h_dec)

    def sampleQ(self, h_enc):
        # TODO epsilon ~ N(0,1) to sample from
        e = None

        # Equation 1. TODO
        mu = None
        # Equation 2. TODO
        log_sigma = None
        sigma = torch.exp(log_sigma)
        
        z = mu + e * sigma

        return z, mu, log_sigma, sigma

    def attn_window(self, h_dec, N):
        # Equation 21.
        params = self.fc_attention(h_dec)
        gx_, gy_, log_sigma_2, log_delta_, log_gamma = params.split(1, 1)

        # Equation 22. TODO 
        gx = None
        # Equation 23
        gy = None
        # Equation 24. TODO
        delta = None
        sigma_2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx, gy, sigma_2, delta, N), gamma

    def filterbank(self, gx, gy, sigma_2, delta, N, epsilon=1e-8):
        grid_i = torch.arange(start=0.0, end=N, device=self.device, requires_grad=True,).view(1, -1)
        
        # Equation 19. TODO
        mu_x = None
        # Equation 20. TODO 
        mu_y = None 

        a = torch.arange(0.0, self.A, device=self.device, requires_grad=True).view(1, 1, -1)
        b = torch.arange(0.0, self.B, device=self.device, requires_grad=True).view(1, 1, -1)

        mu_x = mu_x.view(-1, N, 1)
        mu_y = mu_y.view(-1, N, 1)
        sigma_2 = sigma_2.view(-1, 1, 1)

        # Equations 25 and 26. TODO
        Fx = None
        Fy = None 


        if self.channel == 3:
            Fx = Fx.view(Fx.size(0), 1, Fx.size(1), Fx.size(2))
            Fx = Fx.repeat(1, 3, 1, 1)
            
            Fy = Fy.view(Fy.size(0), 1, Fy.size(1), Fy.size(2))
            Fy = Fy.repeat(1, 3, 1, 1)

        return Fx, Fy

    def loss(self, x):
        self.forward(x)

        # Reconstruction loss.
        Lx = 0 # TODO : BCELoss

        # Latent loss.
        Lz = 0
        for t in range(self.T):
            kl_loss = 0 # TODO
            Lz += kl_loss

        Lz = torch.mean(Lz)
        net_loss = 0 # TODO 

        return net_loss

    def generate(self, num_output):
        self.batch_size = num_output
        h_dec_prev = torch.zeros(num_output, self.dec_size, device=self.device)
        dec_state = torch.zeros(num_output, self.dec_size  , device=self.device)

        for t in range(self.T):
            c_prev = torch.zeros(self.batch_size, self.B*self.A*self.channel, device=self.device) if t == 0 else self.cs[t-1]
            z = torch.randn(self.batch_size, self.z_size, device=self.device)
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec

        imgs = []

        for img in self.cs:
            # The image dimesnion is B x A (According to the DRAW paper).
            img = img.view(-1, self.channel, self.B, self.A)
            imgs.append(vutils.make_grid(torch.sigmoid(img).detach().cpu(), nrow=int(np.sqrt(int(num_output))), padding=1, normalize=True, pad_value=1))

        return imgs