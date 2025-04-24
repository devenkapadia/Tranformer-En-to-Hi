import torch
import torch.nn
import math

class CrossMultiHeadedSelfAttention(torch.nn.Module):
    def __init__(self, dim_model, n_heads,device='cpu'):
        """
        Initialize the Multi-Headed Self-Attention module.

        Args:
            dim_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            device (str): 'cuda' or 'cpu'
        """
        super(CrossMultiHeadedSelfAttention, self).__init__()
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.head_dim = dim_model // n_heads
        self.device = device
        assert (
            self.head_dim * n_heads == dim_model
        ), "dim_model must be divisible by n_heads"

        self.linear_q = torch.nn.ModuleList([torch.nn.Linear(dim_model, self.head_dim) for _ in range(n_heads)])
        self.linear_k = torch.nn.ModuleList([torch.nn.Linear(dim_model, self.head_dim) for _ in range(n_heads)])
        self.linear_v = torch.nn.ModuleList([torch.nn.Linear(dim_model, self.head_dim) for _ in range(n_heads)])

        self.linear_out = torch.nn.Linear(dim_model, dim_model)
        self.dropout = torch.nn.Dropout(0.1)

        self.to(device)

        
    def forward(self, x, encoder_output=None):
        """
        Forward pass for the Multi-Headed Self-Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_model).
        """
        x = x.to(self.device)
        encoder_output = encoder_output.to(self.device)
        batch_size, seq_len, _ = x.size()
        # eg. x(10, 50, 256)
        # Initialize lists to store attention outputs from each head
        attention_outputs = []

        for i in range(self.n_heads):
            q = self.linear_q[i](x)
            k = self.linear_k[i](encoder_output)
            v = self.linear_v[i](encoder_output)
            
            attn_output = self.scaled_dot_product_attention(q, k, v) # Shape: (10, 50, 32)
            attention_outputs.append(attn_output) 
        
        # Concatenate the outputs from all heads
        attention_outputs = torch.cat(attention_outputs, dim=-1) # Shape: (10, 50, 256)
        # Apply the final linear transformation and dropout
        output = self.linear_out(attention_outputs)
        output = self.dropout(output) 
        
        return output # Shape: (10, 50, 256)

    def scaled_dot_product_attention(self, q, k, v):
        """
        Compute the scaled dot-product attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, head_dim).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, head_dim).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, head_dim).

        Returns:
            torch.Tensor: Attention output of shape (batch_size, seq_len, head_dim).
        """
        # Compute the dot product between queries and keys
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply softmax to get attention weights
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        # Compute the attention output as a weighted sum of values
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output