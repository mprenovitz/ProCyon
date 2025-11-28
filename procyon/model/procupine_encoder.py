import torch
import torch.nn as nn
import torch.nn.functional as F

class procupineVAE(nn.Module):
    def __init__(self, rna_dim, hidden_dim, latent_dim):
        super().__init__()

        self.rna_dim = rna_dim
        self.encoder = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rna_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x_hat = None
        mu = None
        logvar = None
        encoder = self.encoder(x)
        mu_input = encoder
        logvar_input = encoder
        mu =self.mu_layer(mu_input)
        logvar = self.logvar_layer(logvar_input)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def reparametrize(self, mu, logvar):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return mu + std * eps

    def mse_loss(self, y, y_pred, mu, logvar):
        return (F.mse_loss(y_pred, y) + torch.sum(-.5 * (1+logvar - (mu ** 2) - torch.exp(logvar)))) /y.shape[0]

    def nb_loss(self, y, y_pred, mu, logvar):
        #main issue here is im just confused how we would use the 
        # binomial distribution since we aren't keeping track the number of trials anywhere
        # i guess this could just be the batch index?
        pass
        

        
