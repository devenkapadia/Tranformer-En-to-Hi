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

def beam_search_predict(input_tokens, transformer, embedding_layer_encoder, embedding_layer_decoder,tokenizer, max_len=32, beam_size=5, device='cpu'):
    # Encode source once
    eng_ids = process_input(input_tokens, tokenizer, max_len)
    src_tensor = torch.tensor([eng_ids], dtype=torch.long, device=device)
    enc = embedding_layer_encoder(src_tensor)
    pos = PositionalEmbedding(max_len, enc.size(-1), device).pe
    enc = enc + pos.expand_as(enc)
    
    # Initialize beam: list of (tokens, logprob)
    beams = [([tokenizer.bos_id()], 0.0)]
    
    for _ in range(max_len):
        all_candidates = []
        for seq, score in beams:
            if seq[-1] == tokenizer.eos_id():
                # already ended; carry forward unchanged
                all_candidates.append((seq, score))
                continue
            
            # Prepare decoder input
            dec_input = torch.tensor([seq], dtype=torch.long, device=device)
            dec_emb = embedding_layer_decoder(dec_input)
            dec_pos = pos[:, :len(seq), :].expand_as(dec_emb)
            dec_enc = dec_emb + dec_pos
            
            with torch.no_grad():
                logits = transformer(enc, dec_enc)[0, -1]  # (vocab,)
                log_probs = torch.log_softmax(logits, dim=-1)
            
            # Expand each beam
            topk_logps, topk_ids = log_probs.topk(beam_size)
            for logp, tok in zip(topk_logps.tolist(), topk_ids.tolist()):
                new_seq = seq + [tok]
                new_score = score + logp
                all_candidates.append((new_seq, new_score))
        
        # Keep top beam_size sequences
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        print(f"Current beams: {beams}")
        # If all beams have ended, stop early
        if all(seq[-1] == tokenizer.eos_id() for seq, _ in beams):
            break
    
    # Return the highest‑scoring completed sequence (or partial if none ended)
    best_seq, best_score = max(beams, key=lambda x: x[1])
    print(f"Best score: {best_score}")
    # strip BOS and EOS
    result_ids = [tok for tok in best_seq if tok not in {tokenizer.bos_id(), tokenizer.eos_id()}]
    return tokenizer.decode_ids(result_ids)

# Example usage
# input_tokens = ["i", "am", "happy"]
while True:
    input_tokens = input("Enter a sentence in English: ").split()
    if not input_tokens:
        break
    translation = beam_search_predict(input_tokens, transformer, embedding_layer_encoder, embedding_layer_decoder, tokenizer, max_len=32, device=device)
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

