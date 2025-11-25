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
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rna_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x_hat = None
        mu = None
        log_var = None

        encoder = self.encoder(x)
        mu_input = encoder
        log_var_input = encoder
        mu =self.mu_layer(mu_input)
        log_var = self.log_var_layer(log_var_input)
        z = self.reparametrize(mu, log_var)
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

    def reparametrize(self, mu, log_var):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * log_var)
        return mu + std * eps

    def mse_loss(self, y, y_pred, mu, log_var):
        return F.mse_loss(y_pred, y) + torch.sum(-.5 * (1+log_var - (mu ** 2) - torch.exp(log_var)))

    def nb_loss(self, y, y_pred, mu, log_var):
        #main issue here is im just confused how we would use the 
        # binomial distribution since we aren't keeping trakc the number of trials anywhere
        pass
        

        
