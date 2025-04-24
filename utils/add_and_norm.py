import torch
import torch.nn

class AddAndNorm(torch.nn.Module):
    def __init__(self, dim_model, device='cpu'):
        """
        Initialize the Add and Norm module.

        Args:
            dim_model (int): Dimension of the model.
        """
        super(AddAndNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(dim_model)
        self.device = device
        self.to(device)

    def forward(self, x, sublayer_output):
        """
        Forward pass for the Add and Norm module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_model).
            sublayer_output (torch.Tensor): Output tensor from the sublayer.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_model).
        """
        x = x.to(self.device)
        sublayer_output = sublayer_output.to(self.device)
        # Add and normalize
        return self.layer_norm(x + sublayer_output)  # Shape: (batch_size, seq_len, dim_model)