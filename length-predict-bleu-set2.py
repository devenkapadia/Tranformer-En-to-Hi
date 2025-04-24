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

data_by_length = {}
for en, hi in test_data:
    pos = en.index(2)
    if pos not in data_by_length:
        data_by_length[pos] = []
    data_by_length[pos].append([en, hi])

# Load the tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.load('en_hi.model')
# english = []
# hindi = []
# for en, hi in test_data:
#     english.append(en)
#     hindi.append(tokenizer.decode_ids(hi))

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

def beam_search_predict(eng_ids, max_len=32, beam_size=5, device='cpu'):
    # Encode source once
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
        # If all beams have ended, stop early
        if all(seq[-1] == tokenizer.eos_id() for seq, _ in beams):
            break
    
    # Return the highest‑scoring completed sequence (or partial if none ended)
    best_seq, best_score = max(beams, key=lambda x: x[1])
    # strip BOS and EOS
    result_ids = [tok for tok in best_seq if tok not in {tokenizer.bos_id(), tokenizer.eos_id()}]
    return result_ids

output_translations = []
bleu_total = 0
#print(len(english))
print(data_by_length.keys())
for key,value in data_by_length.items():
    if key not in [9,23,31,14,13,15,17,12,20]:
        continue
    value=value[:10]
    print("-------------------------------------------------------------")
    print("Length: ", key)
    print("-------------------------------------------------------------")
    for i in range(len(value)):
        curr_english = value[i][0]
        curr_hindi = value[i][1]
        curr_hindi = curr_hindi[:curr_hindi.index(2)+1]
        curr_hindi = tokenizer.decode_ids(curr_hindi)
        translation = beam_search_predict(curr_english, max_len=32, beam_size=5, device=device)
        translation = tokenizer.decode_ids(translation)
        #output_translations.append(translation)
        bleu_score = sentence_bleu([curr_hindi],translation)
        print(bleu_score)
        bleu_total += bleu_score
    print("Bleu Score: ", bleu_total/len(value))


# for i in range(len(english)):
#     curr_english = english[i]
#     curr_hindi = hindi[i]
#     translation = predict(curr_english, max_len=32, device=device)
#     translation = tokenizer.decode_ids(translation)
#     output_translations.append(translation)
#     # print(translation)
#     # print(curr_hindi)
#     # print("-------------------------------------------------------------")
#     bleu_score = sentence_bleu([curr_hindi],translation)
#     print(bleu_score)
#     bleu_total += bleu_score
# print(bleu_total/len(english))

