import numpy as np
import torch
import torch.nn
from transformer import Transformer
from encoder.encoder import Encoder
from decoder.decoder import Decoder
from utils.embedding_layer import EmbeddingLayer
from utils.positional_embedding import PositionalEmbedding

train_data = np.load('train_data.npy', allow_pickle=True)
val_data = np.load('val_data.npy', allow_pickle=True)
train_data = train_data.tolist()
val_data = val_data.tolist()

english = []
hindi = []

for en, hi in train_data:
    english.append(en)
    hindi.append(hi)

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
epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
position = PositionalEmbedding(max_len,dim_model,device).pe
position = position.to(device)
epochs=10
def train():
    # Initialize model components
    embedding_layer_encoder = EmbeddingLayer(vocab_size, dim_model,None, device)
    embedding_layer_decoder = EmbeddingLayer(vocab_size, dim_model,None,  device)
    embedding_layer_encoder.to(device)
    embedding_layer_decoder.to(device)
    encoder = Encoder(dim_model, n_heads, dim_ff, n_encoder_layers, dropout,device)
    decoder = Decoder(dim_model, n_heads, dim_ff, n_decoder_layers, dropout,device)
    transformer = Transformer(encoder, decoder,dim_model,vocab_size,device)
    losses=[]
    # Move model to device
    transformer.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)

    print(sum(p.numel() for p in embedding_layer_encoder.parameters() if p.requires_grad))
    print(sum(p.numel() for p in embedding_layer_decoder.parameters() if p.requires_grad))
    print(sum(p.numel() for p in transformer.parameters() if p.requires_grad))

    # Training loop
    for epoch in range(epochs):
        transformer.train()
        total_loss = 0.0
        print(f"Epoch {epoch + 1}/{epochs}")
        for i in range(0, len(english), batch_size):
            if i + batch_size > len(english):
                break
            print(f"Epoch {epoch} Batch {i // batch_size + 1}/{len(english) // batch_size}")
            eng_batch = torch.tensor(english[i:i+batch_size], dtype=torch.long).to(device)
            encoded_english = embedding_layer_encoder(eng_batch)
            
            position_expanded = position.expand(batch_size, -1, -1)  # [batch_size, seq_len, d_model]
            encoded_with_pos_eng = encoded_english + position_expanded
            
            hin_batch = torch.tensor(hindi[i:i+batch_size], dtype=torch.long).to(device)
            encoded_hindi = embedding_layer_decoder(hin_batch[:, :-1])
            encoded_with_pos_hin = encoded_hindi + position_expanded[:, :hin_batch.shape[1]-1, :]
            
            # Forward pass
            optimizer.zero_grad()
            output = transformer(encoded_with_pos_eng, encoded_with_pos_hin) 
            # Compute loss
            loss = criterion(output.reshape(-1, vocab_size), hin_batch[:, 1:].reshape(-1))  # Exclude first token
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Loss: {total_loss / (len(english) // batch_size)}")
        print(f"Loss: {total_loss / (len(english) // batch_size)}")
        losses.append(total_loss / (len(english) // batch_size))
        # Save the model
        torch.save({
            'transformer_state_dict': transformer.state_dict(),
            'embedding_layer_encoder': embedding_layer_encoder.state_dict(),
            'embedding_layer_decoder': embedding_layer_decoder.state_dict(),
        }, f'translation_model_{epoch}.pth')
        print("Model saved successfully!")
    print(losses)
train()
