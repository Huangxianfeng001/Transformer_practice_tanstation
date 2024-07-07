

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Multi_head_attention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(Multi_head_attention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.linear =clones(nn.Linear(d_model, d_model) , 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = q.size(0)
        q,k,v =[ l(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2) for l, x in zip(self.linear, [q, k, v])]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill( mask==0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, v)
        #x （batch_size, n_head, seq_len, d_k），先转成(batch_size, seq_len, n_head * d_k)，即(batch_size, seq_len, d_model)
        #通过线性转化输出
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        return  self.linear[-1](x)