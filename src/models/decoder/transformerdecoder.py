import torch
import torch.nn as nn
import math
from torch import Tensor

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        nhead: int,
        n_classes: int,
        max_len: int=5760 // 2,
    ):
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, 1)

        self.pe = PositionalEncoding(
            hidden_size, 
            dropout=0.0, 
            max_len=max_len, 
            batch_first=True,)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dropout=dropout, 
            batch_first=True, 
            dim_feedforward=128,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=num_layers
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        x = self.conv(x)  # (batch_size, n_channels, n_timesteps)
        x = x.transpose(1, 2)  # (batch_size, n_timesteps, n_channels)
        x = self.pe(x)
        x = self.transformer_encoder(x)
        x = self.linear(x)  # (batch_size, n_timesteps, n_classes)

        return x



class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        batch_first: bool = True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe[: x.size(0), :]
        else:
            x = x + self.pe[: x.size(0)]
        return self.dropout(x)