import torch
import torch.nn

class FeedForwardLayer(torch.nn.Module):
    def __init__(self, dim_model, dim_ff, dropout=0.1,device='cpu'):
        """
        Initialize the Feed Forward Layer.

        Args:
            dim_model (int): Dimension of the model.
            dim_ff (int): Dimension of the feed forward layer.
            dropout (float): Dropout rate.
            device (str): 'cuda' or 'cpu'
        """
        super(FeedForwardLayer, self).__init__()
        self.device = device

        self.linear1 = torch.nn.Linear(dim_model, dim_ff)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_ff, dim_model)
        self.to(device)


    def forward(self, x):
        """
        Forward pass for the Feed Forward Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_model).
        """
        # Apply ReLU activation and dropout
        x = x.to(self.device)
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        
        # Apply final linear transformation
        return self.linear2(x)  # Shape: (batch_size, seq_len, dim_model)