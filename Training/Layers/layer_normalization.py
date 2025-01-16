import torch.nn as nn
import torch

class LayerNormalization(nn.Module):

    def __init__(self, d_model , eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward (self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.std(dim = -1,keepdim = True)
        return self.alpha * (x-mean)/torch.sqrt(var+ self.eps) + self.beta
