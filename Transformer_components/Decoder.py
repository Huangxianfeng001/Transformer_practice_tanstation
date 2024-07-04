import torch
import torch.nn as nn

from self_attention import Multi_head_attention
from Encoder import LayerNorm, FeedForward, PositionalEncoding

class Decoderlayer(nn.Module):
    def __init__(self, d_model, n_head, d_inner_hid, dropout):
        super(Decoderlayer, self).__init__()
        self.atten = Multi_head_attention(d_model, n_head, dropout)
        self.layernorm = LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, d_inner_hid, dropout)

    def forward(self, x, Encoder_KV, mask = None):
        # x: [batch_size, seq_len, emb_dim]
        # mask: [batch_size, seq_len]
        z = self.attention(x, x, x, mask)
        x = self.layer_norm(x+z)

        #second mult-attention block
        z = self.attention(x, Encoder_KV, Encoder_KV, mask)
        x = self.layer_norm(x+z)

        z2 = self.feed_forward(x)
        return self.layer_norm(x+z2)
        

class Decoder(nn.Module):
    def __init__(self, d_model, n_head, d_inner_hid, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([ Decoderlayer(d_model, n_head, d_inner_hid, dropout) for _ in range(6)])
        self.pe = PositionalEncoding(d_model, dropout)
    def forward(self, x, EN_KV, mask):
        # x: [batch_size, seq_len, emb_dim]
        # mask: [batch_size, seq_len]
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, EN_KV, mask)
        return x

        
