import sentencepiece as spm
import torch
import os

# Step 1: Train the SentencePiece model with a vocab size of 128,000
def train_sentencepiece(input_file, model_prefix, vocab_size=800):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        pad_id=0,        # [PAD]
        unk_id=1,        # [UNK]
        bos_id=2,        # [SOS]
        eos_id=3,        # [EOS]
        user_defined_symbols=["[SOS]", "[EOS]"]  # Define special tokens
    )

# Step 2: Tokenize the text file and load SentencePiece model
def load_tokenizer(model_prefix):
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp

# Print the vocabulary with token IDs
def print_vocab(tokenizer):
    print("Vocabulary:")
    for id in range(tokenizer.get_piece_size()):
        print(f"ID {id}: {tokenizer.id_to_piece(id)}")

# Step 3: Create embedding layer with PyTorch
class BPETokenEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(BPETokenEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, input_ids):
        return self.embedding(input_ids)

# Main function
def main():
    # Input text file for training SentencePiece (replace with your dataset path)
    input_file = 'translation_dict1.txt'
    model_prefix = 'tokenizer_model'
    vocab_size = 800
    embedding_dim = 512

    # Check if the tokenizer model already exists to avoid retraining
    if not os.path.isfile(f"{model_prefix}.model"):
        print("Training SentencePiece model...")
        train_sentencepiece(input_file, model_prefix, vocab_size)

    # Load the trained SentencePiece model
    tokenizer = load_tokenizer(model_prefix)

    # Print the vocabulary with token IDs
    print_vocab(tokenizer)

    # Example text to tokenize
    text = "Hi"

    # Tokenize text into token IDs
    token_ids = tokenizer.encode(text, out_type=int)
    print("Token IDs:", token_ids)

    # Create embedding layer with vocab size and embedding dimension
    embedding_layer = BPETokenEmbedding(vocab_size, embedding_dim)
    
    # Convert token IDs to a PyTorch tensor and pass through the embedding layer
    input_tensor = torch.tensor(token_ids).unsqueeze(0)  # Add batch dimension
    embeddings = embedding_layer(input_tensor)
    
    print("Embeddings shape:", embeddings.shape)
    print("Embeddings:", embeddings)

if __name__ == "__main__":
    main()
