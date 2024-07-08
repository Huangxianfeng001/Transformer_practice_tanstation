import numpy as np
import torch
import matplotlib.pyplot as plt
from Transformer_components.Model   import *
from Transformer_components.Data_process  import *
from Transformer_components.Decoder   import *
from Transformer_components.Encoder   import *
from Transformer_components.self_attention   import *
from Transformer_components.Feedforward   import *
import sys

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#data_prepare show plot
# batch_size = 128

# Embeddings_dim = 64
# src_vocab_size = 10000
# tgt_vocab_size = 10000
# max_len = 50
# d_model = 64
# nhead = 8
# num_encoder_layers = 6
# num_decoder_layers = 6
# dim_feedforward = 2048
# dropout = 0.1

# max_seq_len = 100
# seq_len = 20


#show the postion encoding plot
# pe = PositionalEncoding(Embeddings_dim, 0, max_seq_len)
# positional_encoding = pe(torch.zeros(1, seq_len, Embeddings_dim, device=DEVICE))
# plt.figure()
# sns.heatmap(positional_encoding.squeeze().to("cpu"))
# plt.xlabel("i")
# plt.ylabel("pos")
# plt.show()

# plt.figure()
# y = positional_encoding.to("cpu").numpy()
# plt.plot(np.arange(seq_len), y[0, :, 0 : 64 : 8], ".")
# plt.legend(["dim %d" % p for p in [0, 7, 15, 31, 63]])
# plt.show()


# #plot the subsequent mask size = 20
# plt.figure(figsize=(5, 5))
# plt.imshow(subsequent_mask(20)[0])
# plt.show()


# Label smoothing的例子
# crit = LabelSmoothing(5, 0, 0.4)  # 设定一个ϵ=0.4
# predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
#                              [0, 0.2, 0.7, 0.1, 0],
#                              [0, 0.2, 0.7, 0.1, 0]])
# v = crit(Variable(predict.log()),
#          Variable(torch.LongTensor([2, 1, 0])))

# Show the target distributions expected by the system.
# print(crit.true_dist)
# plt.imshow(crit.true_dist)
# plt.show()


# opts = [NoamOptim(512, 1, 4000, None),
#         NoamOptim(512, 1, 8000, None),
#         NoamOptim(256, 1, 4000, None)]
# plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
# plt.legend(["512:4000", "512:8000", "256:4000"])
# plt.show()


def run_epoch(data, model, loss_compute, epoch):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.
    for i, batch in enumerate(data):
        out = model(batch.src, batch.src_mask, batch.trg, batch.trg_mask)
        # print(out.size())
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch: %d, Batch Step: %d Loss: %f Tokens per Sec: %f" %
                  (epoch, i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
            
    return total_loss / total_tokens
        
def train(data, model, loss_compute, epochs):
    best_loss = 100000
    for epoch in range(epochs):
        model.train()
        print('>>>>> training')
        dev_loss = run_epoch(data.train_data, model, loss_compute, epoch)
        print('<<<<< taining loss: %f' % dev_loss)
        model.eval()

        print('>>>>> Evaluate')
        dev_loss = run_epoch(data.val_data, model, loss_compute, epoch)
        print('<<<<< Evaluate loss: %f' % dev_loss)

        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print('best loss is updated: %f\n' % best_loss)
            print('****** Save model done... ******\n')

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    传入一个训练好的模型，对指定数据进行预测
    """
    # 先用encoder进行encode
    # memory = model.Encoder(src, src_mask)
    memory = model.embedding_src(src)
    memory = model.Encoder(memory, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    ys1 = model.embedding_tgt(ys)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.Decoder(Variable(ys1),
                            memory,
                            src_mask,
                            Variable(subsequent_mask(ys1.size(1)).type_as(memory.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(memory.data).fill_(next_word)], dim=1)
    return ys

def test_result(data, data_test, model, loss_compute, nums_test = 10):
    model.load_state_dict(torch.load('best_model.pt'))

    model.eval()


    print('>>>>> Evaluate')
    with torch.no_grad():
        for i in range(nums_test):
            print(len(data.en_index_dict))
            print(data.en_index_dict.items)
            en_sent =" ".join([data1.en_index_dict.get(w) for w in data1.en_train[i]])
            print("eng sentens is:\n" + en_sent)

            cn_sent = " ".join([data1.cn_index_dict[w] for w in data1.cn_train[i]])
            print("cn sentens is:\n" + cn_sent)

            src = torch.from_numpy(np.array(data1.en_train[i])).long().to(DEVICE)

            src = src.unsqueeze(0)
            # 设置attention mask
            src_mask = (src != 0).unsqueeze(-2)

            out_prob = greedy_decode(model, src, src_mask, max_len=50, start_symbol=data.cn_dict['BOS'])
            print(out_prob.size())
            print(out_prob[0, 10])
            translation = []
            for j in range(1, out_prob.size(1)):
                # 获取当前下标的输出字符
                sym = data.cn_index_dict[out_prob[0, j].item()]
                # 如果输出字符不为'EOS'终止符，则添加到当前语句的翻译结果列表
                if sym != 'EOS':
                    translation.append(sym)
                # 否则终止遍历
                else:
                    break
            cn_sent_pred = " ".join(translation)
            print("cn sentens pred is:\n" + cn_sent_pred)

Train_file_path = 'en-cn/train.txt'
Val_file_path = 'en-cn/dev.txt'
Test_file_path = 'en-cn/test_mini.txt'
batch_size = 64

data = Data_process(Train_file_path, Val_file_path, batch_size)
src_vocab = len(data.en_dict)
tgt_vocab = len(data.cn_dict)
print("src_vocab vector length is %d" % src_vocab)
print("tgt_vocab vector length is %d" % tgt_vocab)

Embeddings_dim = 512
d_inner_hid = 1024
n_head = 8
dropout = 0.1

model = Transformer_model(src_vocab, tgt_vocab, Embeddings_dim, n_head , d_inner_hid, dropout).to(DEVICE)

criterion = LabelSmoothing(tgt_vocab, padding_idx=0, smoothing=0.0) 
optimizer = NoamOptim(Embeddings_dim, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

loss = SimpleLossCompute(model.generator, criterion, optimizer)
# train(data, model, loss, epochs=10)

data1 = Data_process(Test_file_path, Val_file_path, batch_size)
test_result(data, data1, model, loss, nums_test=10)