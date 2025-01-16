import re
from collections import defaultdict
import numpy as np

class BPE:
    def __init__(self, num_merges):
        self.num_merges = num_merges
        self.vocab = {}
    
    def get_vocab(self, text):
        # Initialize vocabulary with character counts
        for word in text.split():
            word = ' '.join(list(word)) + ' </w>'  # Add end of word token
            if word in self.vocab:
                self.vocab[word] += 1
            else:
                self.vocab[word] = 1

    def get_stats(self):
        pairs = defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair):
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in self.vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = self.vocab[word]
        self.vocab = new_vocab

    def fit(self, text):
        self.get_vocab(text)
        for _ in range(self.num_merges):
            pairs = self.get_stats()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merge_vocab(best_pair)

    def encode(self, text):
        tokens = []
        for word in text.split():
            word = ' '.join(list(word)) + ' </w>'
            if word in self.vocab:
                tokens.append(word)
            else:
                tokens.append("OOV: " + word)
        return tokens

def rotary_embedding(embeddings):
    """
    Apply rotary positional embedding to the input embeddings.
    
    Args:
        embeddings: A numpy array of shape (seq_len, embed_dim).
    
    Returns:
        A numpy array with rotary positional embeddings applied.
    """
    seq_len, embed_dim = embeddings.shape
    # Create rotary positional embeddings
    theta = np.arange(seq_len)[:, None] / (10000 ** (np.arange(0, embed_dim, 2) / embed_dim))
    sin = np.sin(theta)
    cos = np.cos(theta)
    
    # Apply rotary embeddings
    rotary_emb = np.zeros_like(embeddings)
    rotary_emb[:, 0::2] = embeddings[:, 0::2] * cos + embeddings[:, 1::2] * sin
    rotary_emb[:, 1::2] = embeddings[:, 1::2] * cos - embeddings[:, 0::2] * sin
    
    return rotary_emb

# Example usage
if __name__ == "__main__":
    # Read text from a file
    file_path = "C:/Users/admin/Documents/intern aicte LLM/rotatory positional embeddings/translation_dict.txt"
    with open(file_path, 'r',encoding= 'utf-8') as file:
        text = file.read()

    # Initialize BPE and fit to the text
    bpe = BPE(num_merges=10)
    bpe.fit(text)

    # Encode the text to get BPE tokens
    tokens = bpe.encode(text)
    print("BPE Tokens:", tokens)

    # Create embeddings for the BPE tokens
    num_tokens = min(len(tokens), 10)  # Limit to 10 tokens for the sequence length
    embed_dim = 4096  # Embedding dimension

    # Create a fixed embedding matrix based on token indices
    embeddings = np.zeros((10, embed_dim))
    for i in range(num_tokens):
        embeddings[i] = np.arange(embed_dim) + (i * 1000)  # Example fixed embedding

    # Apply rotary embeddings
    rotary_emb = rotary_embedding(embeddings)
    print("Original Embeddings:\n", embeddings)
    print("Rotary Positional Embeddings:\n", rotary_emb)