from numpy import SHIFT_UNDERFLOW
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd


data_path = "spam.csv"
# text preprocessing 一般在这里写个function完成

# 完成数据集类
class MyDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv(data_path)

    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        text = self.df.iloc[index]['text'].strip()
        label = self.df.iloc[index]['label']
        return label, text
    
    def __len__(self):
        # 返回数据的总数量
        return self.df.shape[0]



my_dataset = MyDataset()
data_loader = DataLoader(dataset=my_dataset, batch_size=2, shuffle=True)


if __name__ == '__main__':
    for index, (label, text) in enumerate(data_loader):
        print("label: {}, text: {}".format(label, text))
        print("*"*100)


        
# if __name__ == '__main__':
#     print(my_dataset[0]) # this due to "__getitem__" method from the class
#     print(len(my_dataset)) # this due to "__len__" method from the class

# token -> num -> vector
# torch.nn.Embedding(num_embeddings, embedding_dim)

"""
[batch_size, seq_len] ---> [batch_size, seq_len, embedding_dim]
"""