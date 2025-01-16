import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super(RotaryPositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, embeddings: torch.Tensor):
        # Handle both 3D and 4D input tensors
        # print(embeddings.shape)
        if len(embeddings.shape) == 3:
            batch_size, seq_len, embed_dim = embeddings.shape
            n_heads = None
            head_dim = embed_dim
        elif len(embeddings.shape) == 4:
            batch_size, n_heads, seq_len, head_dim = embeddings.shape
        else:
            raise ValueError(f"Expected 3D or 4D input, got {len(embeddings.shape)}D input")

        if head_dim != self.embed_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embed_dim}, got {head_dim}")

        # Compute the rotary embeddings for the seq_len and head_dim
        theta = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1) / (
            10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

        # Convert sin and cos to have compatible shapes for broadcasting
        sin = torch.sin(theta).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim//2)
        cos = torch.cos(theta).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim//2)

        # If 4D (multi-head attention), apply the rotary embedding per head
        if len(embeddings.shape) == 4:
            rotary_emb = torch.zeros_like(embeddings)
            rotary_emb[:, :, :, 0::2] = embeddings[:, :, :, 0::2] * cos + embeddings[:, :, :, 1::2] * sin
            rotary_emb[:, :, :, 1::2] = embeddings[:, :, :, 1::2] * cos - embeddings[:, :, :, 0::2] * sin
        else:  # If 3D, apply rotary embedding without the heads dimension
            rotary_emb = torch.zeros_like(embeddings)
            rotary_emb[:, :, 0::2] = embeddings[:, :, 0::2] * cos + embeddings[:, :, 1::2] * sin
            rotary_emb[:, :, 1::2] = embeddings[:, :, 1::2] * cos - embeddings[:, :, 0::2] * sin

            # print("rr",rotary_emb.size())

        return rotary_emb
