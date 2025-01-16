from torch import nn
from .layer_normalization import LayerNormalization
from .feedforward import FeedForward
from .multi_head import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self,d_model, n_head, hidden,p =0.1):
        super().__init__()
        self.d_k = d_model // n_head
        self.att =MultiHeadAttention(d_model = d_model, n_head=n_head)
        self.norm1 = LayerNormalization(d_model)
        self.drop1= nn.Dropout(p)
        self.cross = MultiHeadAttention(d_model = d_model, n_head=n_head)
        self.norm2 = LayerNormalization(d_model)
        self.drop2= nn.Dropout(p)
        self.ffn = FeedForward(d_model = d_model, hidden = hidden)
        self.norm3 = LayerNormalization(d_model)
        self.drop3= nn.Dropout(p)

    def forward(self, x, enc=None, mask=None):
        res = self.att(q=x, k=x, v=x, mask=mask)
        res = self.drop1(res)
        res = self.norm1(res + x)

        if enc is not None:  
            res_ = res
            res = self.cross(q=res, k=enc, v=enc)
            res = self.drop2(res)
            res = self.norm2(res_ + res)

        res_ = res
        res = self.ffn(res)
        res = self.drop3(res)
        output = self.norm3(res + res_) 

        return output
