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
