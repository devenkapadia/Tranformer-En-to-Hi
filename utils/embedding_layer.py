import torch
import torch.nn

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, vocab_size, dim_model, padding_idx=None,  device='cpu'):
        super(EmbeddingLayer, self).__init__()
        self.device = device
        self.embedding = torch.nn.Embedding(vocab_size, dim_model, padding_idx=padding_idx)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.embedding(x)
