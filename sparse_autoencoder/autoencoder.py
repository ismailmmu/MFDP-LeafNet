import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def sparse_loss_function(reconstructed, original, encoded, sparsity_lambda):
    mse_loss = F.mse_loss(reconstructed, original)
    l1_loss = sparsity_lambda * torch.mean(torch.abs(encoded))  # L1 sparsity constraint
    return mse_loss + l1_loss


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded