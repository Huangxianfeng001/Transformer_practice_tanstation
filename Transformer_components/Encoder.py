import torch
import torch.nn as nn
from .self_attention import Multi_head_attention
from .Feedforward import FeedForward
from torch.autograd import Variable
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        # α、β分别初始化为1、0
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # 沿词向量方向计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # 沿词向量和语句序列方向计算均值和方差
        # mean = x.mean(dim=[-2, -1], keepdim=True)
        # std = x.std(dim=[-2, -1], keepdim=True)
        # 归一化
        x = (x - mean) / torch.sqrt(std ** 2 + self.eps)
        return self.a_2 * x + self.b_2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 位置编码矩阵，维度[max_len, embedding_dim]
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        # 单词位置
        position = torch.arange(0.0, max_len, device=DEVICE)
        position.unsqueeze_(1)
        # 使用exp和log实现幂运算
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=DEVICE) * (- math.log(1e4) / d_model))
        div_term.unsqueeze_(0)
        # 计算单词位置沿词向量维度的纹理值
        pe[:, 0 : : 2] = torch.sin(torch.mm(position, div_term))
        pe[:, 1 : : 2] = torch.cos(torch.mm(position, div_term))
        # 增加批次维度，[1, max_len, embedding_dim]
        pe.unsqueeze_(0)
        # 将位置编码矩阵注册为buffer(不参加训练)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个批次中语句所有词向量与位置编码相加
        # 注意，位置编码不参与训练，因此设置requires_grad=False
        x += Variable(self.pe[:, : x.size(1), :], requires_grad=False)
        return self.dropout(x)

class Encoderlayer(nn.Module):
    def __init__(self,d_model, n_head, d_inner_hid,dropout):
        super(Encoderlayer, self).__init__()
        self.attention = Multi_head_attention(d_model, n_head, dropout)
        self.layer_norm = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model,d_inner_hid,dropout)

    def forward(self, x, mask):
        # x: [batch_size, seq_len, emb_dim]
        # mask: [batch_size, seq_len]
        # print(x.size())
        # print(mask.size())
        z = self.attention(x, x, x, mask)
        x = self.layer_norm(x+z)
        z2 = self.feed_forward(x)
        return self.layer_norm(x+z2)
        

class Encoder(nn.Module):
    def __init__(self,d_model, n_head, d_inner_hid, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([ Encoderlayer(d_model, n_head, d_inner_hid, dropout) for _ in range(6)])
        self.pe = PositionalEncoding(d_model, dropout)
    def forward(self, x, mask):
        # x: [batch_size, seq_len, emb_dim]
        # mask: [batch_size, seq_len]
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, mask)
        return x
