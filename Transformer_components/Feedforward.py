import torch
import torch.nn as nn
from torch import Tensor
class FeedForward(nn.Module):
    def __init__(self, d_model, d_inner_hid=512, dropout=0.1):
        super(FeedForward, self).__init__()
        self.seq = nn.Sequential(
            nn.LayerNorm(d_model, 1e-6),
            nn.Linear(d_model, d_inner_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner_hid, d_model),
            nn.Dropout(dropout)
            )
    
    def forward(self, inputs: Tensor):
        return self.seq(inputs)