import torch
import torch.nn
from encoder.encoder_layer import EncoderLayer

class Encoder(torch.nn.Module):
    def __init__(self, dim_model, n_heads, dim_ff, n_layers, dropout=0.1,device='cpu'):
        """
        Initialize the Encoder.

        Args:
            dim_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            dim_ff (int): Dimension of the feed forward layer.
            n_layers (int): Number of encoder layers.
            dropout (float): Dropout rate.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        super(Encoder, self).__init__()
        self.device = device
        self.layers = torch.nn.ModuleList(
            [EncoderLayer(dim_model, n_heads, dim_ff, dropout,device) for _ in range(n_layers)]
        )
        self.to(device)

    def forward(self, x):
        """
        Forward pass for the Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_model).
        """
        x = x.to(self.device)
        for layer in self.layers:
            layer.to(self.device)
            x = layer(x)

        
        return x
