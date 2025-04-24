import torch
import torch.nn
from decoder.cross_multi_headed_self_attention import CrossMultiHeadedSelfAttention
from decoder.masked_multi_headed_self_attention import MaskedMultiHeadedSelfAttention
from utils.feedforward_layer import FeedForwardLayer
from utils.add_and_norm import AddAndNorm


class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_model, n_heads, dim_ff, dropout=0.1,device='cpu'):
        """
        Initialize the Decoder Layer.

        Args:
            dim_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            dim_ff (int): Dimension of the feed forward layer.
            dropout (float): Dropout rate.
            device (str): Device to run on ('cpu' or 'cuda')
        """
        super(DecoderLayer, self).__init__()
        self.device = device

        self.masked_self_attention = MaskedMultiHeadedSelfAttention(dim_model, n_heads, device)
        self.add_and_norm1 = AddAndNorm(dim_model, device)
        
        self.cross_attention = CrossMultiHeadedSelfAttention(dim_model, n_heads, device)
        self.add_and_norm2 = AddAndNorm(dim_model, device)
        
        self.feed_forward = FeedForwardLayer(dim_model, dim_ff, dropout, device)
        self.add_and_norm3 = AddAndNorm(dim_model, device)

        self.to(device)

    def forward(self, x, encoder_output):
        """
        Forward pass for the Decoder Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_model).
            memory (torch.Tensor): Memory tensor from the encoder of shape (batch_size, seq_len, dim_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_model).
        """
        x = x.to(self.device)
        encoder_output = encoder_output.to(self.device)
        
        # Apply self-attention and add & norm
        attn_output1 = self.masked_self_attention(x)
        x = self.add_and_norm1(x, attn_output1)

        # Apply cross-attention and add & norm
        attn_output2 = self.cross_attention(x, encoder_output)
        x = self.add_and_norm2(x, attn_output2)

        # Apply feed forward and add & norm
        ff_output = self.feed_forward(x)
        x = self.add_and_norm3(x, ff_output)

        return x  # Shape: (batch_size, seq_len, dim_model)