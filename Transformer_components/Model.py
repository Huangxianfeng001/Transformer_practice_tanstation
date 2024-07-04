import torch.nn as nn
import torch
from Encoder import Encoder
from Decoder import Decoder

class Generator(nn.Module):
    """
    解码器输出经线性变换和softmax函数映射为下一时刻预测单词的概率分布
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x的词向量（需要乘以math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)

class Transformer_model(nn.Module):
    def __init__(self, src_vovab, tgr_vovab, d_model, n_head, d_inner_hid, dropout):
        super(Transformer_model, self).__init__()
        self.Encoder = Encoder(d_model, n_head, d_inner_hid, dropout)
        self.Decoder = Decoder(d_model, n_head, d_inner_hid, dropout)
        self.generator = Generator(d_model, tgr_vovab)
        self.embedding_src = Embeddings(d_model, src_vovab)
        self.embedding_tgt = Embeddings(d_model, tgr_vovab)

    def forward(self, src, src_mask, tgt, tgt_mask):
        src_x = self.embedding_src(src)
        tgt_x = self.embedding_tgt(tgt)
        enc_output = self.Encoder(src_x, src_mask)
        tgt_mask = (tgt != 0).unsqueeze(-2)
        dec_output = self.Decoder(tgt, enc_output, tgt_mask)
        output = self.generator(dec_output)
        return output