import os
import torch
from torch import nn
from model.transformer_decoder import TransformerDecoder
from model.layers.embed import TokenEmbedding
from model.tokenizer.train_tokenizer import train_sentencepiece_multilingual
from model.tokenizer.tokenizer_utils import load_tokenizer
from data import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Input text file for training SentencePiece
input_file = './data/output_part1.txt'
model_prefix = 'multilingual_tokenizer'
vocab_size = 10000
embedding_dim = 2048

# Check if the tokenizer model already exists to avoid retraining
if not os.path.isfile(f"{model_prefix}.model"):
    print("Training SentencePiece model...")
    file_paths = ['data/output_part1.txt']
    train_sentencepiece_multilingual(file_paths, model_prefix=model_prefix, vocab_size=vocab_size)

# Load the trained SentencePiece model
tokenizer = load_tokenizer(model_prefix)

# Example text to tokenize
text = "You taught me language"

# Tokenize text into token IDs
token_ids = tokenizer.encode(text, out_type=int)

# Create embedding layer with vocab size and embedding dimension
embedding_layer = TokenEmbedding(vocab_size, embedding_dim)

# Convert token IDs to a PyTorch tensor and pass through the embedding layer
input_tensor = torch.tensor([token_ids], dtype=torch.long)  # Add batch dimension
embeddings = embedding_layer(input_tensor)

# Instantiate the TransformerDecoder
seq_len = 100  # Maximum sequence length
d_model = 1920  # Embedding dimension
n_head = 96  # Number of attention heads
hidden_dim = 1920  # Dimension of feed-forward network
dec_voc_size = 10000  # Vocabulary size of the SentencePiece model
n_layers = 32  # Number of layers in the decoder

# Instantiate the model
model = TransformerDecoder(d_model, n_head, hidden_dim, dec_voc_size, n_layers)

total_params = count_parameters(model)
print(f'Total number of trainable parameters: {total_params}')

# Example input sentence
input_sentence = "You taught me language"

# Tokenize the input sentence
input_ids = tokenizer.encode(input_sentence, out_type=int)
input_tensor = torch.tensor([input_ids], dtype=torch.long)  # Convert to tensor, batch size = 1

# Run the input through the model to generate output
with torch.no_grad():  # No gradient tracking during inference
    output_probs = model(input_tensor)  # Shape: [1, seq_len, vocab_size]

# Function to generate the output sequence by sampling or taking the argmax of the probabilities
def generate_text(model, tokenizer, input_ids, max_length=50):
    model.eval()
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    for _ in range(max_length - len(input_ids)):
        with torch.no_grad():
            output_probs = model(input_tensor)  # Forward pass
            next_token_logits = output_probs[:, -1, :]  # Get logits for the next token
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()  # Choose the token with the highest probability
            input_ids.append(next_token_id)
            input_tensor = torch.tensor([input_ids], dtype=torch.long)  # Update input tensor with new token

            # Break if EOS token is generated
            if hasattr(tokenizer, "eos_id") and next_token_id == tokenizer.eos_id():
                break

    # Convert token IDs back to text (ensure input_ids is a list of integers)
    generated_text = tokenizer.decode(input_ids)
    return generated_text

# Start generation from the tokenized input
try:
    generated_text = generate_text(model, tokenizer, input_ids, max_length=50)
    print("Generated Text:", generated_text)
except Exception as e:
    print("Error during text generation:", str(e))
