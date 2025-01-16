from torch import nn
from .layer_normalization import LayerNormalization
from .feedforward import FeedForward
from .multi_head import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self,d_model, n_head, hidden):
        super().__init__()
        self.d_k = d_model // n_head
        self.att =MultiHeadAttention(d_model = d_model, n_head=n_head)
        self.norm = LayerNormalization(d_model)
        self.ffn = FeedForward(d_model = d_model, hidden = hidden)
        self.norm_ = LayerNormalization(d_model)

    def forward(self, x, mask):
        x_ = x
        res = self.att(q=x, k=x, v=x, mask = mask)
        res = self.norm(res + x)
        res_ = res
        res = self.ffn(res)
        output = self.norm_(res + res_) 

        return output
