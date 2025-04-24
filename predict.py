import torch
import sentencepiece as spm
import re
from transformer import Transformer
from encoder.encoder import Encoder
from decoder.decoder import Decoder
from utils.embedding_layer import EmbeddingLayer
from utils.positional_embedding import PositionalEmbedding

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

# Initialize model components
embedding_layer_encoder = EmbeddingLayer(vocab_size, dim_model, None, device)
embedding_layer_decoder = EmbeddingLayer(vocab_size, dim_model, None, device)
encoder = Encoder(dim_model, n_heads, dim_ff, n_encoder_layers, dropout, device)
decoder = Decoder(dim_model, n_heads, dim_ff, n_decoder_layers, dropout, device)
transformer = Transformer(encoder, decoder, dim_model, vocab_size, device)

# Load the saved model state
checkpoint = torch.load('translation_model_20.pth', map_location=device)
transformer.load_state_dict(checkpoint['transformer_state_dict'])
embedding_layer_encoder.load_state_dict(checkpoint['embedding_layer_encoder'])
embedding_layer_decoder.load_state_dict(checkpoint['embedding_layer_decoder'])

# Set the model to evaluation mode
transformer.eval()

def clean_input(tokens):
    """Clean English tokens by lowercasing and removing non-alphanumeric characters."""
    cleaned_tokens = [re.sub(r'[^a-z0-9\s]', '', t.lower()) for t in tokens if t]
    return ' '.join(cleaned_tokens)

def process_input(input_tokens, tokenizer, max_len=32):
    """Convert input tokens to numerical IDs with BOS, EOS, and padding."""
    cleaned_input = clean_input(input_tokens)
    eng_ids = tokenizer.encode_as_ids(cleaned_input)
    
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    pad_id = tokenizer.pad_id()
    if pad_id == -1:
        pad_id = 0
    
    # Truncate to max_len - 2, add BOS and EOS, pad to max_len
    eng_ids = eng_ids[:max_len - 2]
    eng_ids = [bos_id] + eng_ids + [eos_id]
    if len(eng_ids) < max_len:
        eng_ids += [pad_id] * (max_len - len(eng_ids))
    
    return eng_ids

def predict(input_tokens, transformer, embedding_layer_encoder, embedding_layer_decoder, tokenizer, max_len=32, device=device):
    """Generate Hindi translation from English input tokens."""
    # Process input
    eng_ids = process_input(input_tokens, tokenizer, max_len)
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

    # Decode output IDs to Hindi sentence
    print(output_sequence)
    hindi_sentence = tokenizer.decode_ids(output_sequence)
    return hindi_sentence

# Example usage
# input_tokens = ["i", "am", "happy"]
while True:
    input_tokens = input("Enter a sentence in English: ").split()
    if not input_tokens:
        break
    translation = predict(input_tokens, transformer, embedding_layer_encoder, embedding_layer_decoder, tokenizer, max_len=32, device=device)
    print(f"Translation: {translation}")
# input_tokens = ["sports", "are", "my", "favorite", "hobby"]
# translation = predict(input_tokens, transformer, embedding_layer_encoder, embedding_layer_decoder, tokenizer, max_len=32, device=device)
# print(f"Translation: {translation}")
#Translation: मुझे खुशी है कि मैं इस बात पर खुशी कर रहा हूं कि मैं इसे एक ीकरण के लिए एक अच्छा अवसर मिला ।
#10 epoch Translation: जब मैं एक संदेश को एक संदेश भेज रहा है तो मैं एक अधिकारी को एक अधिसूचना देता है ।
#1 epoch Translation: यह एक बार फिर से एक बार फिर से एक बार फिर से एक बार फिर से एक बार फिर से एक नया था ।

#Translation: भारत मेरी पसंदीदा देश है ।

# Translation: मैं गहन ज्ञान को प्यार करता हूँ

# Translation: जो भारत के राष्ट्रपति हैं
# Translation: मेरा नाम भारत है ।
# "their is a tall tree on the top of the hill" -> Translation: उनके पहाड़ी के ऊपर एक ऊंचे पेड़ है । 
# "how are you today my friend" --> आज मेरे दोस्त कैसे हैं ?
