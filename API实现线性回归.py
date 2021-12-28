import torch
from torch import optim
from torch.optim import SGD
from torch import nn

# GPU配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# all inputs and model object needs to do: .to(device)

#0  准备数据
x = torch.rand([500, 1]).to(device)
y_true = 3*x + 0.8
y_true = y_true.to(device)


#1  定义模型
class MyLinear(nn.Module):
    def __init__(self):
        super(MyLinear, self).__init__()
        self.linear = nn.Linear(1,1)


    def forward(self, x):
        out = self.linear(x)
        return out


#2  实例化模型， 优化器实例化， loss实例化
my_linear = MyLinear().to(device)
optimizer = SGD(my_linear.parameters(), 0.001)
loss_fn = nn.MSELoss()


#3 循环， 精选梯度下降，参数的更新
for i in range(10000000):
    #得到预测值
    y_pred = my_linear(x)
    # print(y_pred)
    # break
    loss = loss_fn(y_pred, y_true)
    #梯度置为0
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    #参数的更新
    optimizer.step()
    if i%100 == 0:
        params = list(my_linear.parameters())
        # y = mx + b
        print("loss: {}, m: {}, b: {}".format(loss.item(), params[0].item(), params[1].item()))
        print("\n")


#4 model评估
my_linear.eval()
predict = my_linear(x)
predict = predict.cpu.detach().numpy()

"""
执行结果需要和cpu的tensor计算的时候要把cuda的tensor转成cpu的tensor
"""