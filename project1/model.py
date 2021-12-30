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
        self.fc = nn.Linear(hidden_size * 2, 2)#([hidden_size * 2 -> (256), 2]), because bidirectional, fw has 128 and bw has 128, 128 + 128 is just 256
    
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
        # shape: [4, 128]
        output_bw = h_n[-1, :, :] # 反向最后一次的输出
        # shape: [4, 128]
        output = torch.cat([output_fw, output_bw], dim=-1) #[batch_size, hidden_size*2]
        # shape:[4, 256]
        out = self.fc(output) #可以考虑添加一个新的全连接层作为输出层， 激活函数处理
        # shape:[16, 2]
        return F.log_softmax(out, dim=-1)



model = NeuralNet().to(device)
optimizer = Adam(model.parameters(), 0.001)
if os.path.exists("models\model.pkl"):
    model.load_state_dict(torch.load("models\model.pkl"))
    optimizer.load_state_dict(torch.load("models\optimizer.pkl"))

def train(epoch):
    path = r"C:\Users\45323\OneDrive\桌面\新python文件夹\pytorch\project1\IMDB Dataset.csv"
    dataloader = get_dataloader(path, batch_size=batch_size, train=True)
    for idx, (input, target) in enumerate(dataloader):
        input = input.to(device)
        target = target.to(device)
        #梯度归0
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Index: {idx}, Loss: {loss.item()}")

        if idx%100 == 0:
            torch.save(model.state_dict(),"models\model.pkl")
            torch.save(optimizer.state_dict(),"models\optimizer.pkl")



def eval():
    loss_list  =[]
    acc_list = []
    path = r"C:\Users\45323\OneDrive\桌面\新python文件夹\pytorch\project1\IMDB Dataset.csv"
    dataloader = get_dataloader(path, batch_size=test_batch_size, train=False)
    for idx, (input, target) in enumerate(dataloader):
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
    # for i in range(10):
    #     start = time.time()
    #     train(i)
    #     stop = time.time()
    #     print(f"Training Elapsed for epoch {i} : {stop - start}s")

# NOTE: max_len = 200
# NOTE: embed_dim = 100
# NOTE: len(ws) = 7002

# check if pytorch is using GPU:
# 1. go to CMD``
# 2. enter: nvidia-smi
# 3. see if GPU was used, don't trust the stupid TASK MANAGER



#def test():
#     test_loss = 0
#     correct = 0
#     model.eval()
#     path = r"C:\Users\45323\OneDrive\桌面\新python文件夹\pytorch\project1\IMDB Dataset.csv"
#     test_dataloader = get_dataloader(path, batch_size=batch_size, train=False)
#     with torch.no_grad():
#         for idx, (input, target) in enumerate(test_dataloader):
#             input = input.to(device)
#             target = target.to(device)
#             output = model(input)
#             pred = torch.max(output, dim=-1, keepdim=False)[-1]
#             test_loss += F.nll_loss(output, target, reduction="sum")
#             correct = pred.eq(target.data).sum()
#         test_loss = test_loss/len(test_dataloader.dataset)
#         print("\nTest set Avg. loss: {:.4f}, Accuracy: {:.2f}%".format(test_loss, 
#                                                                    100.0 * correct/len(test_dataloader.dataset)))