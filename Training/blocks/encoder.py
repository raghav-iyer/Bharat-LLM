from ..layers.encoder_layer import EncoderLayer
from ..layers.decoder_layer import DecoderLayer
from model.layers.embed import TokenEmbedding
from torch import nn

class Encoder(nn.Module):
    def __init__(self, d_model, n_head, hidden_dim, dec_voc_size, n_layers):
        super(Encoder, self).__init__()
        self.embedding = TokenEmbedding(vocab_size=dec_voc_size, embedding_dim=d_model)  
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, hidden_dim) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, dec_voc_size)  

    def forward(self, input, mask):
        x = self.embedding(input) 
        for layer in self.layers:
            x = layer(x=x,mask = mask)
        output = self.linear(x)
        return output
