import math
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        d_k = k.size(-1)  
        k_t = k.transpose(2,3) 

        # print("mul",q.shape, k_t.shape,k.shape)

        score = torch.matmul(q , k_t) / math.sqrt(d_k)

        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))

        score = self.softmax(score)
        # print("v",score.size(),v.size())
        return torch.matmul(score , v), score
