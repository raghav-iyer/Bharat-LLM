from ..layers.decoder_layer import DecoderLayer
from ..layers.embed import TokenEmbedding
from torch import nn

class Decoder(nn.Module):
    def __init__(self, d_model, n_head, hidden_dim, dec_voc_size, n_layers):
        super(Decoder, self).__init__()
        self.embedding = TokenEmbedding(vocab_size=dec_voc_size, embedding_dim=d_model)  
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, hidden_dim) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, dec_voc_size)  

    def forward(self, enc, output, mask):
        x = self.embedding(output) 

        for layer in self.layers:
            x = layer(x=x, enc=enc, mask=mask)

        output = self.linear(x)
        return output
