"""
定义模型
"""
import torch
from torch import optim
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import dataloader
from dataset import DataLoader, get_dataloader
import torch.nn.functional as F
from lib import ws, max_len, embed_dim, hidden_size, bidirectional, num_layers, dropout, device, batch_size, test_batch_size
import os
import numpy as np
import tqdm

# NOTE: for timing
import time

"""
模型优化方案:
添加一个新的全连接层作为输出层, 激活函数处理
把双向的LSTM的output传给一个单向的LSTM再进行处理
"""

# 检查GPU配置
print("I am using the device: ", device)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.embedding = nn.Embedding(len(ws), embed_dim)#([7002, 100])\
        # 加入LSTM
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout)
        self.neural_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, 128), #([hidden_size * 2 -> (256), 128]), because bidirectional, fw has 128 and bw has 128, 128 + 128 is just 256
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
    
    def forward(self, input): # input must be tensor not list
        """
        param input: [batch_size, max_len]
        return:
        """
        x = self.embedding(input) #embedding操作, 前形状: [batch_size, max_len]  后形状: [batch_size, max_len, 100]
        # shape:[16, 200, 100]
        x, (h_n, c_n) = self.lstm(x)
        # -> x:[batch_size, max_len, 2*hidden_size,], h_n(c_n too):[num_layers*2, batch_size, hidden_size]
        # 获取两个方向最后一次的output, 进行concat
        output_fw = h_n[-2, :, :] # 正向最后一次的输出
        # shape: [512, 128] # 512 is batch_size
        output_bw = h_n[-1, :, :] # 反向最后一次的输出
        # shape: [512, 128] # 512 is batch_size
        output = torch.cat([output_fw, output_bw], dim=-1) #[batch_size, hidden_size*2], do concatenation on [2]
        # shape:[512, 256]
        out = self.neural_layers(output) # shape: [512, 2]
        return F.softmax(out, dim=-1)



model = NeuralNet().to(device)
optimizer = Adam(model.parameters(), 0.001)
loss_func = nn.CrossEntropyLoss()
if os.path.exists("models\model.pkl"):
    model.load_state_dict(torch.load("models\model.pkl"))
    optimizer.load_state_dict(torch.load("models\optimizer.pkl"))

def train(epoch):
    path = r"C:\Users\45323\OneDrive\桌面\新python文件夹\pytorch\project1\IMDB Dataset.csv"
    dataloader = get_dataloader(path, batch_size=batch_size, train=True)
    for idx, (input, target) in tqdm.tqdm(enumerate(dataloader),
                                          total=len(dataloader.dataset)/batch_size,
                                          ascii=True,
                                          desc="训练"):
        input = input.to(device)
        target = target.to(device)
        #梯度归0
        optimizer.zero_grad()
        output = model(input)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        if idx%100 == 0:
            torch.save(model.state_dict(),"models\model.pkl")
            torch.save(optimizer.state_dict(),"models\optimizer.pkl")
    print(f"Epoch: {epoch}, Loss: {loss.item()}")



def eval():
    loss_list  =[]
    acc_list = []
    path = r"C:\Users\45323\OneDrive\桌面\新python文件夹\pytorch\project1\IMDB Dataset.csv"
    dataloader = get_dataloader(path, batch_size=test_batch_size, train=False)
    for idx, (input, target) in tqdm.tqdm(enumerate(dataloader),
                                          total=len(dataloader.dataset)/test_batch_size,
                                          ascii=True,
                                          desc="测试"):
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss.cpu().item())
            #计算准确率
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc.cpu().item())
    print("total Loss: {}, Accuracy: {}".format(np.mean(loss_list), np.mean(acc_list)))



#NOTE: 参数 在 lib.py
if __name__ == "__main__":
    # NOTE: for evaluation
    eval()

    # # NOTE: training
    # for i in range(12): # epoch = 10
    #     start = time.time()
    #     train(i)
    #     stop = time.time()
    #     print(f"Training Elapsed for epoch {i} : {stop - start}s")

# NOTE: max_len = 200
# NOTE: embed_dim = 100
# NOTE: len(ws) = 7002