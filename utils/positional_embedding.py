import torch
import math # Can use math or torch functions

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, max_len, dim_model,device='cpu'):
        super(PositionalEmbedding, self).__init__()
        self.max_len = max_len
        self.dim_model = dim_model
        self.device = device
        self.pe = self.generate_positional_embedding(max_len, dim_model).to(device)

    def generate_positional_embedding(self, max_len, dim_model):
        """
        Generate positional embeddings using nested for loops.

        Args:
            max_len (int): Maximum length of the sequence.
            dim_model (int): Dimension of the model.

        Returns:
            torch.Tensor: Positional embeddings of shape (1, max_len, dim_model).
        """
        # Initialize the positional embedding tensor with zeros
        pe = torch.zeros(max_len, dim_model,device=self.device)

        # Iterate through each position (sequence element)
        for pos in range(max_len):
            # Iterate through each dimension of the embedding
            for i in range(dim_model):
                # Calculate the division term based on the dimension pair index (floor(i/2))
                # Use float division and ensure tensor type for pow
                exponent = torch.tensor((2 * (i // 2)) / dim_model, dtype=torch.float, device=self.device)
                div_term = torch.pow(torch.tensor(10000.0, device=self.device), exponent)  # Use 10000.0 for float calculation

                # Calculate the argument for sin/cos
                # Ensure pos is treated as float for division
                angle_arg = torch.tensor(pos / div_term, dtype=torch.float, device=self.device)


                # Apply sin to even indices
                if i % 2 == 0:
                    pe[pos, i] = torch.sin(angle_arg)
                # Apply cos to odd indices
                else:
                    pe[pos, i] = torch.cos(angle_arg)

        # Add a batch dimension at the beginning, as in the original return statement
        return pe.unsqueeze(0)  # Shape: (1, max_len, dim_model)

    def forward(self):
        return self.pe
