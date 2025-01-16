from .layers.decoder_layer import DecoderLayer
from .layers.rotary import RotaryPositionalEmbedding
from tokenizer.bpe import BPETokenEmbedding
from torch import nn
import torch

class TransformerDecoder(nn.Module):
    def __init__(self, seq_len, d_model, n_head, hidden_dim, dec_voc_size, n_layers):
        super(TransformerDecoder, self).__init__()
        self.embedding = BPETokenEmbedding(vocab_size=dec_voc_size, embedding_dim=d_model)  
        self.rotary_emb = RotaryPositionalEmbedding(d_model)  
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, hidden_dim) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, dec_voc_size)  
        self.sm = nn.Softmax(dim=-1)

    def generate_square_subsequent_mask(self, size):
        mask = torch.tril(torch.ones(size, size))
        return mask.unsqueeze(0).unsqueeze(0)  

    def forward(self, input):
        seq_len = input.size(1)  
        x = self.embedding(input) 
        mask = self.generate_square_subsequent_mask(seq_len).to(input.device)  

        enc = self.layers[0](x=x, mask=mask)
        for layer in self.layers[1:]:
            x = layer(x=x, enc=enc, mask=mask)

        output = self.linear(x)
        probs = self.sm(output)
        return probs