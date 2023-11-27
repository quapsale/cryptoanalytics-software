"""
File: gru.py
Description: GRU model.
File Created: 06/04/2022
Python Version: 3.9
"""

# Imports
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# Send to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# GRU Model
class GRU(pl.LightningModule):
    """
    Constructor.
    """

    def __init__(self, n_features=4, hidden_units=100, n_layers=10, lr=1e-3, dev=device):
        super(GRU, self).__init__()
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.lr = lr
        self.dev = dev
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_units,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=0.1
        )
        self.linear = nn.Linear(self.hidden_units, 1)

    # Forward Pass
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_units).requires_grad_().to(self.dev)
        out, hn = self.gru(x, h0.detach())
        out = self.linear(out[:, -1, :])
        return out

    # Training step
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.unsqueeze(1)
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    # Validation step
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.unsqueeze(1)
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    # Prediction step
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)
        return y_hat

    # Optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
