import torch
import torch.nn
from decoder.decoder_layer import DecoderLayer

class Decoder(torch.nn.Module):
    def __init__(self, dim_model, n_heads, dim_ff, n_layers, dropout=0.1,device='cpu'):
        """
        Initialize the Decoder.

        Args:
            dim_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            dim_ff (int): Dimension of the feed forward layer.
            n_layers (int): Number of decoder layers.
            dropout (float): Dropout rate.
            device (str): 'cuda' or 'cpu'.
        """
        super(Decoder, self).__init__()
        self.device = device
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(dim_model, n_heads, dim_ff, dropout,device) for _ in range(n_layers)]
        )
        self.to(device)

        
    def forward(self, x, encoder_output):
        """
        Forward pass for the Decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_model).
            encoder_output (torch.Tensor): Memory tensor from the encoder of shape (batch_size, seq_len, dim_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_model).
        """
        x = x.to(self.device)
        encoder_output = encoder_output.to(self.device)
        
        for layer in self.layers:
            x = layer(x, encoder_output)
        
        return x  # Shape: (batch_size, seq_len, dim_model)