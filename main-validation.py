import numpy as np
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
from transformer import Transformer
from encoder.encoder import Encoder
from decoder.decoder import Decoder
from utils.embedding_layer import EmbeddingLayer
from utils.positional_embedding import PositionalEmbedding
import matplotlib.pyplot as plt

# Load data
train_data = np.load('train_data.npy', allow_pickle=True).tolist()
val_data = np.load('val_data.npy', allow_pickle=True).tolist()

# Define the dataset class
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        en, hi = self.data[idx]
        en_tensor = torch.tensor(en, dtype=torch.long)
        hi_tensor = torch.tensor(hi, dtype=torch.long)
        return en_tensor, hi_tensor

# Hyperparameters
dim_model = 128
n_heads = 8
dim_ff = 1024
dropout = 0.1
vocab_size = 30000
n_encoder_layers = 5
n_decoder_layers = 5
max_len = 32
batch_size = 64
learning_rate = 0.001
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize positional embedding
position = PositionalEmbedding(max_len, dim_model, device).pe.to(device)

# Create datasets and dataloaders
train_dataset = TranslationDataset(train_data)
val_dataset = TranslationDataset(val_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

def train():
    # Initialize model components
    embedding_layer_encoder = EmbeddingLayer(vocab_size, dim_model, None, device).to(device)
    embedding_layer_decoder = EmbeddingLayer(vocab_size, dim_model, None, device).to(device)
    encoder = Encoder(dim_model, n_heads, dim_ff, n_encoder_layers, dropout, device)
    decoder = Decoder(dim_model, n_heads, dim_ff, n_decoder_layers, dropout, device)
    transformer = Transformer(encoder, decoder, dim_model, vocab_size, device).to(device)

    #check point
    checkpoint = torch.load('translation_model_5.pth', map_location=device)
    transformer.load_state_dict(checkpoint['transformer_state_dict'])
    embedding_layer_encoder.load_state_dict(checkpoint['embedding_layer_encoder'])
    embedding_layer_decoder.load_state_dict(checkpoint['embedding_layer_decoder'])


    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)

    # Print parameter counts
    print(f"Encoder Embedding Params: {sum(p.numel() for p in embedding_layer_encoder.parameters() if p.requires_grad)}")
    print(f"Decoder Embedding Params: {sum(p.numel() for p in embedding_layer_decoder.parameters() if p.requires_grad)}")
    print(f"Transformer Params: {sum(p.numel() for p in transformer.parameters() if p.requires_grad)}")

    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(6, epochs + 1):
        transformer.train()
        total_loss = 0.0
        print(f"Epoch {epoch}/{epochs}")
        for batch_idx, (eng_batch, hin_batch) in enumerate(train_loader):
            eng_batch = eng_batch.to(device)
            hin_batch = hin_batch.to(device)
            if (batch_idx+1) % 1000 == 0:
                print(f"Epoch {epoch} Batch {batch_idx + 1}/{len(train_loader)}")
            # Encoder forward pass
            encoded_english = embedding_layer_encoder(eng_batch)
            position_expanded = position.expand(eng_batch.size(0), -1, -1)
            encoded_with_pos_eng = encoded_english + position_expanded

            # Decoder forward pass
            encoded_hindi = embedding_layer_decoder(hin_batch[:, :-1])
            encoded_with_pos_hin = encoded_hindi + position_expanded[:, :hin_batch.size(1) - 1, :]

            # Training step
            optimizer.zero_grad()
            output = transformer(encoded_with_pos_eng, encoded_with_pos_hin)
            loss = criterion(output.reshape(-1, vocab_size), hin_batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")
        train_losses.append(avg_train_loss)

        # Validation loop
        transformer.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for val_eng_batch, val_hin_batch in val_loader:
                val_eng_batch = val_eng_batch.to(device)
                val_hin_batch = val_hin_batch.to(device)

                # Encoder forward pass
                val_encoded_en = embedding_layer_encoder(val_eng_batch)
                val_position_expanded = position.expand(val_eng_batch.size(0), -1, -1)
                val_encoded_with_pos_en = val_encoded_en + val_position_expanded

                # Decoder forward pass
                val_encoded_hi = embedding_layer_decoder(val_hin_batch[:, :-1])
                val_encoded_with_pos_hi = val_encoded_hi + val_position_expanded[:, :val_hin_batch.size(1) - 1, :]

                # Validation step
                val_output = transformer(val_encoded_with_pos_en, val_encoded_with_pos_hi)
                val_loss = criterion(val_output.reshape(-1, vocab_size), val_hin_batch[:, 1:].reshape(-1))
                val_total_loss += val_loss.item()

        avg_val_loss = val_total_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        val_losses.append(avg_val_loss)

        # Save the model
        torch.save({
            'transformer_state_dict': transformer.state_dict(),
            'embedding_layer_encoder': embedding_layer_encoder.state_dict(),
            'embedding_layer_decoder': embedding_layer_decoder.state_dict(),
        }, f'translation_model_{epoch}.pth')
        print("Model saved successfully!")

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.show()

    print(f"Final Training Loss: {avg_train_loss:.4f}")
    print(f"Final Validation Loss: {avg_val_loss:.4f}")

train()