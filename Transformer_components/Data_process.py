import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from nltk import word_tokenize
from torch.autograd import Variable

from langconv.converter import LanguageConverter
from langconv.language.zh import zh_cn, zh_tw  # zh_hk also supported

lc_cn = LanguageConverter.from_language(zh_cn)  # target variant set to zh-cn
lc_tw = LanguageConverter.from_language(zh_tw)  # target variant set to zh-tw

print(lc_cn.convert('人人生而自由，在尊嚴和權利上一律平等。他們賦有理性和良心，並應以兄弟關係的精神相對待。'))
# Expected:          人人生而自由，在尊严和权利上一律平等。他们赋有理性和良心，并应以兄弟关系的精神相对待。
print(lc_tw.convert('人人生而自由，在尊严和权利上一律平等。他们赋有理性和良心，并应以兄弟关系的精神相对待。'))
# Expected:   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cht_to_chs(sent):
    sent = lc_cn.convert(sent)
    sent.encode("utf-8")
    return sent

def seq_padding(X, padding=0):
    """
    按批次（batch）对数据填充、长度对齐
    """
    # 计算该批次各条样本语句长度
    L = [len(x) for x in X]
    # 获取该批次样本中语句长度最大值
    ML = max(L)
    # 遍历该批次样本，如果语句长度小于最大长度，则用padding填充
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

def subsequent_mask(size):
    "Mask out subsequent positions."
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """
    批次类
        1. 输入序列（源）
        2. 输出序列（目标）
        3. 构造掩码
    """

    def __init__(self, src, trg=None, pad=0):
        # 将输入、输出单词id表示的数据规范成整数类型
        src = torch.from_numpy(src).to(device).long()
        trg = torch.from_numpy(trg).to(device).long()
        self.src = src
        # 对于当前输入的语句非空部分进行判断，bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对解码器使用的目标语句进行掩码
        if trg is not None:
            # 解码器使用的目标输入部分
            self.trg = trg[:, : -1] # 去除最后一列
            # 解码器训练时应预测输出的目标结果
            self.trg_y = trg[:, 1:] #去除第一列的SOS
            # 将目标输入部分进行注意力掩码
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # print(self.trg_y.size())
            # print(self.trg_mask.size())
            # 将应输出的目标结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # 掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class Data_process:
    def __init__(self, path_tr, path_va, batch_size, max_words=50000):
        self.en, self.cn = self.load_data(path_tr)
        
        self.en_val_ori, self.cn_val_ori = self.load_data(path_va)

        self.en_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.en, max_words)
        self.cn_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.cn, max_words)
        
        self.en_train, self.cn_train = self.word2id(self.en, self.cn, self.en_dict, self.cn_dict)
        self.en_val, self.cn_val = self.word2id(self.en_val_ori, self.cn_val_ori, self.en_dict, self.cn_dict)
        
        self.train_data = self.split_batch(self.en_train, self.cn_train, batch_size)
        self.val_data = self.split_batch(self.en_val, self.cn_val, batch_size)

    def load_data(self, file_path):
        cn = []
        en = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for lines in file.readlines():
                sent_en, sen_cn = lines.strip().split('\t')
                sent_cn = cht_to_chs(sen_cn)
                sent_en = sent_en.lower()
                sent_en = ["BOS"] + word_tokenize(sent_en) + ["EOS"]
                sent_cn = ["BOS"] + [char for char in sent_cn] + ["EOS"]
                cn.append(sent_cn)
                en.append(sent_en)
        return en, cn
    

    def build_dict(self, sentences, max_words=50000):
        word_count = Counter([word for sent in sentences for word in sent])
        ls = word_count.most_common(int(max_words))
        total_words = len(ls) + 2 # account for UNK and PAD
        word_dict = {w[0]: index+2 for index, w in enumerate(ls)} # 0: <UNK>, 1: <PAD>
        word_dict['UNK'] = 1
        word_dict['PAD'] = 0
        
        index_dict = {v:k for k,v in word_dict.items()}
        return word_dict, total_words, index_dict
    
    def word2id(self, en, cn, en_dict, cn_dict, sort=True):
        """
        将英文、中文单词列表转为单词索引列表
        `sort=True`表示以英文语句长度排序，以便按批次填充时，同批次语句填充尽量少
        """
        length = len(en)
        # 单词映射为索引
        out_en_ids = [[en_dict.get(word, 1) for word in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(word, 1) for word in sent] for sent in cn]

        # 按照语句长度排序
        def len_argsort(seq):
            """
            传入一系列语句数据(分好词的列表形式)，
            按照语句长度排序后，返回排序后原来各语句在数据中的索引下标
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 按相同顺序对中文、英文样本排序
        if sort:
            # 以英文语句长度排序
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[idx] for idx in sorted_index]
            out_cn_ids = [out_cn_ids[idx] for idx in sorted_index]
        return out_en_ids, out_cn_ids

    def split_batch(self, en, cn, batch_size, shuffle=True):
        """
        划分批次
        `shuffle=True`表示对各批次顺序随机打乱
        """
        # 每隔batch_size取一个索引作为后续batch的起始索引
        idx_list = np.arange(0, len(en), batch_size)
        # 起始索引随机打乱
        if shuffle:
            np.random.shuffle(idx_list)
        # 存放所有批次的语句索引
        batch_indexs = []
        for idx in idx_list:
            """
            形如[array([4, 5, 6, 7]), 
                 array([0, 1, 2, 3]), 
                 array([8, 9, 10, 11]),
                 ...]
            """
            # 起始索引最大的批次可能发生越界，要限定其索引
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))
        # 构建批次列表
        batches = []
        for batch_index in batch_indexs:
            # 按当前批次的样本索引采样
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            # 对当前批次中所有语句填充、对齐长度
            # 维度为：batch_size * 当前批次中语句的最大长度
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            # 将当前批次添加到批次列表
            # Batch类用于实现注意力掩码
            batches.append(Batch(batch_en, batch_cn))
        return batches      