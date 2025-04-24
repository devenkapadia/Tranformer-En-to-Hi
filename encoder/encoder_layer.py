import torch
import torch.nn 
from encoder.multi_headed_self_attention import MultiHeadedSelfAttention
from utils.feedforward_layer import FeedForwardLayer
from utils.add_and_norm import AddAndNorm

class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_model, n_heads, dim_ff, dropout=0.1,device='cpu'):
        """
        Initialize the Encoder Layer.

        Args:
            dim_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            dim_ff (int): Dimension of the feed forward layer.
            dropout (float): Dropout rate.
            device (str): Device to run on ('cuda' or 'cpu').

        """
        super(EncoderLayer, self).__init__()
        self.device = device

        self.self_attention = MultiHeadedSelfAttention(dim_model, n_heads,device)
        self.add_and_norm1 = AddAndNorm(dim_model,device)

        self.feed_forward = FeedForwardLayer(dim_model, dim_ff, dropout,device)
        self.add_and_norm2 = AddAndNorm(dim_model,device)

        self.to(device)


    def forward(self, x):
        """
        Forward pass for the Encoder Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_model).
        """
        x = x.to(self.device)

        # Apply self-attention and add & norm
        attn_output = self.self_attention(x)
        x = self.add_and_norm1(x, attn_output)

        # Apply feed forward and add & norm
        ff_output = self.feed_forward(x)
        x = self.add_and_norm2(x, ff_output)

        return x  # Shape: (batch_size, seq_len, dim_model)