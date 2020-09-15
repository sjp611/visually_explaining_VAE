import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

class VAE_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, mu, log_var, rec_c, kl_c):
        batch_size = logits.shape[0]
        
        rec_loss = F.binary_cross_entropy(logits, labels, reduction = 'sum') * rec_c
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0) * kl_c

        total_loss = rec_loss + kld_loss

        return total_loss, rec_loss, kld_loss

class VAE_model(nn.Module):
    def __init__(self, args):
        super(VAE_model, self).__init__()

        self.feature_in = args.feature_in
        self.feature_num = args.feature_num
        self.feature_num_2 = args.feature_num * 2 
        self.latent_dim = args.latent_dim

        self.encoder1 = nn.Sequential( # 28
            nn.Conv2d(self.feature_in, self.feature_num, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(self.feature_num),
            nn.ReLU(inplace=True),
        )

        self.encoder2 = nn.Sequential( # 14
            nn.Conv2d(self.feature_num, self.feature_num_2, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(self.feature_num_2),
            nn.ReLU(inplace=True),
        )

        self.encoder3 = nn.Sequential( # 7
            nn.Conv2d(self.feature_num_2, self.feature_num_2, (4, 4), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(self.feature_num_2),
            nn.ReLU(inplace=True),
        )

        self.mean = nn.Sequential( 
            nn.Linear(self.feature_num_2 * 4 * 4, self.latent_dim)
        )

        self.variance = nn.Sequential( 
            nn.Linear(self.feature_num_2 * 4 * 4, self.latent_dim)
        )


        self.decoder1 = nn.Sequential( 
            nn.Linear(self.latent_dim, self.feature_num_2 * 4 * 4) 
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_num_2, self.feature_num_2, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(self.feature_num_2),
            nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_num_2, self.feature_num, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(self.feature_num),
            nn.ReLU(inplace=True)
        )
            
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_num, self.feature_in, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Sigmoid()
        )


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu

        return z

    def encode(self, x):
        x = self.encoder1(x)
        #print(x.shape)
        x = self.encoder2(x)
        #print(x.shape)
        x = self.encoder3(x)
        #print(x.shape)
        latent = x.view(x.size(0), -1)
        #print(latent.shape)
        mu = self.mean(latent)
        #print(mu.shape)
        log_var = self.variance(latent)
        #print(log_var.shape)

        return mu, log_var

    def decode(self, x):
        x  = self.decoder1(x)
        #print(x.shape)
        x = x.view(x.size(0), self.feature_num_2, 4, 4)
        #print(x.shape)
        x  = self.decoder2(x)
        x = x[:, :, :7, :7]
        #print(x.shape)
        x  = self.decoder3(x)
        #print(x.shape)
        x_hat  = self.decoder4(x)

        return x_hat

    def forward(self, x, train=True):
        mu, log_var = self.encode(x)

        if train:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu

        x_hat = self.decode(z)

        return x_hat, mu, log_var


