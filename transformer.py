import torch
import torch.nn

class Transformer(torch.nn.Module):
    def __init__(self, encoder, decoder, dim_model, vocab_size,device='cpu'):
        """
        Initialize the Transformer model.

        Args:
            encoder (torch.nn.Module): Encoder module.
            decoder (torch.nn.Module): Decoder module.
        """
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc_out = torch.nn.Linear(dim_model, vocab_size)

        self.to(device)

    def forward(self, src, tgt):
        """
        Forward pass for the Transformer model.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            tgt (torch.Tensor): Target input tensor of shape (batch_size, tgt_seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_seq_len, dim_model).
        """
        encoder_output = self.encoder(src)
        decoder_output = self.decoder(tgt, encoder_output)
        output = self.fc_out(decoder_output)
        return output  # Shape: (batch_size, tgt_seq_len, vocab_size)