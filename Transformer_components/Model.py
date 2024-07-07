import torch.nn as nn
import torch
from .Encoder import *
from .Decoder import *
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
    
class NoamOptim(torch.optim.Optimizer):
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            
        self._rate = rate
        self.optimizer.step()
            
    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
def get_std_opt(model):
    return NoamOptim(model.src_embed[0].d_model, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class LabelSmoothing(nn.Module):
    """
    标签平滑
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    def __init__(self, generator, criterion, opt):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()

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
        
        dec_output = self.Decoder(tgt_x, enc_output, src_mask, tgt_mask)
        # output = self.generator(dec_output)
        return dec_output