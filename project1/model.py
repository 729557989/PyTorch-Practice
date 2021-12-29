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
from lib import ws, max_len, embed_dim

# NOTE: for timing
import time



# GPU配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("I am using the device: ", device)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.embedding = nn.Embedding(len(ws), embed_dim)#([7002, 100])
        self.fc = nn.Linear(max_len*embed_dim, 2)        #([20*100, 2])
    
    def forward(self, input): # input must be tensor not list
        """
        param input: [batch_size, max_len]
        return:
        """
        x = self.embedding(input) #embedding操作, 前形状: [batch_size*embed_dim]  后形状: [batch_size, max_len, 100]
        # shape:[256, 20, 100]
        x = x.view([-1, max_len*embed_dim])#([-1, 20*100]) -> 不用管batch_size, 用 -1 即可
        # shape:[256, 2000]
        out = self.fc(x)
        # shape:[256, 2]
        return F.log_softmax(out, dim=-1)



model = NeuralNet().to(device)
optimizer = Adam(model.parameters(), 0.001)
def train():
    path = r"C:\Users\45323\OneDrive\桌面\新python文件夹\pytorch\project1\IMDB Dataset.csv"
    dataloader = get_dataloader(path, batch_size=4, train=True)
    for idx, (input, target) in enumerate(dataloader):
        input = input.to(device)
        target = target.to(device)
        #梯度归0
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        print(loss.item())

def test():
    test_loss = 0
    correct = 0
    model.eval()
    path = r"C:\Users\45323\OneDrive\桌面\新python文件夹\pytorch\project1\IMDB Dataset.csv"
    test_dataloader = get_dataloader(path, batch_size=16, train=False)
    with torch.no_grad():
        for idx, (input, target) in enumerate(test_dataloader):
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            pred = torch.max(output, dim=-1, keepdim=False)[-1]
            test_loss += F.nll_loss(output, target, reduction="sum")
            correct = pred.eq(target.data).sum()
        test_loss = test_loss/len(test_dataloader.dataset)
        print("\nTest set Avg. loss: {:.4f}, Accuracy: {:.2f}%".format(test_loss, 
                                                                   100.0 * correct/len(test_dataloader.dataset)))


if __name__ == "__main__":
    for i in range(1):
        start = time.time()
        train()
        stop = time.time()
        print(f"Training Elapsed for epoch {i} : {stop - start}s")

# NOTE: max_len = 20
# NOTE: embed_dim = 100
# NOTE: len(ws) = 7002

# check if pytorch is using GPU:
# 1. go to CMD
# 2. enter: nvidia-smi
# 3. see if GPU was used, don't trust the stupid TASK MANAGER