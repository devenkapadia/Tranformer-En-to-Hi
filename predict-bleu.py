import torch
import sentencepiece as spm
import re
from transformer import Transformer
from encoder.encoder import Encoder
from decoder.decoder import Decoder
from utils.embedding_layer import EmbeddingLayer
from utils.positional_embedding import PositionalEmbedding
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

test_data = np.load('test_data.npy', allow_pickle=True)
test_data = test_data.tolist()


# Define constants matching training
dim_model = 128
n_heads = 8
dim_ff = 1024
dropout = 0.1
vocab_size = 30000
n_encoder_layers = 5
n_decoder_layers = 5
max_len = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.load('en_hi.model')
english = []
hindi = []
for en, hi in test_data:
    english.append(en)
    curr_hindi = hi[:hi.index(2)+1]
    hindi.append(tokenizer.decode_ids(curr_hindi))

# Initialize model components
embedding_layer_encoder = EmbeddingLayer(vocab_size, dim_model, None, device)
embedding_layer_decoder = EmbeddingLayer(vocab_size, dim_model, None, device)
encoder = Encoder(dim_model, n_heads, dim_ff, n_encoder_layers, dropout, device)
decoder = Decoder(dim_model, n_heads, dim_ff, n_decoder_layers, dropout, device)
transformer = Transformer(encoder, decoder, dim_model, vocab_size, device)

# Load the saved model state
checkpoint = torch.load('translation_model_11.pth', map_location=device)
transformer.load_state_dict(checkpoint['transformer_state_dict'])
embedding_layer_encoder.load_state_dict(checkpoint['embedding_layer_encoder'])
embedding_layer_decoder.load_state_dict(checkpoint['embedding_layer_decoder'])

# Set the model to evaluation mode
transformer.eval()

def predict(eng_ids, max_len=32, device=device):
    # Process input
    eng_tensor = torch.tensor([eng_ids], dtype=torch.long).to(device)
    
    # Encode input with embeddings and positional encodings
    encoded_english = embedding_layer_encoder(eng_tensor)
    position = PositionalEmbedding(max_len, dim_model, device).pe
    position_expanded = position.expand(1, -1, -1)
    encoded_with_pos_eng = encoded_english + position_expanded

    # Initialize decoder input with BOS
    decoder_input = [tokenizer.bos_id()]
    output_sequence = []

    # Autoregressive prediction loop
    for _ in range(max_len):
        decoder_tensor = torch.tensor([decoder_input], dtype=torch.long).to(device)
        encoded_decoder = embedding_layer_decoder(decoder_tensor)
        position_expanded_dec = position.expand(1, -1, -1)[:, :len(decoder_input), :]
        encoded_with_pos_dec = encoded_decoder + position_expanded_dec
        
        # Generate next token
        with torch.no_grad():
            output = transformer(encoded_with_pos_eng, encoded_with_pos_dec)
        next_token_logits = output[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        
        # Stop if EOS is generated
        if next_token_id == tokenizer.eos_id():
            break
        
        output_sequence.append(next_token_id)
        decoder_input.append(next_token_id)

    return output_sequence

output_translations = []
bleu_total = 0
print(len(english))

for i in range(len(english)):
    curr_english = english[i]
    curr_hindi = hindi[i]
    translation = predict(curr_english, max_len=32, device=device)
    translation = tokenizer.decode_ids(translation)
    output_translations.append(translation)
    # print(translation)
    # print(curr_hindi)
    # print("-------------------------------------------------------------")
    bleu_score = sentence_bleu([curr_hindi],translation)
    print(bleu_score)
    bleu_total += bleu_score
print(bleu_total/len(english))

