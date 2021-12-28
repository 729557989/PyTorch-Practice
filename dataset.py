from numpy import dtype
import torch
from torch.utils.data import Dataset, DataLoader, dataloader
import torch.nn.functional as F
import pandas as pd
import os
import re
from lib import ws

# 1 数据集的准备

class ImdbDataset(Dataset):
    def __init__(self, path, train=True, shuffle=False, clean=False, split_size=0.7):
        self.train = train
        self.data_path = path
        self.dataset = pd.read_csv(self.data_path)
        self.train_df = self.dataset[:int(self.dataset.shape[0]*split_size)] # self.dataset.shape[0] * 0.7 = 35000
        self.test_df = self.dataset[int(self.dataset.shape[0]*split_size):]  # self.dataset.shape[0] * 0.3 = 15000
        
        if shuffle == True:
            self.dataset = self.dataset.sample(frac=1)

    def __getitem__(self, index):
        label = self.train_df.iloc[index]['sentiment'] if self.train else self.test_df.iloc[index]['sentiment']
        label = 1 if label == 'positive' else 0 # 0 == negative, 1 == positive
        text = self.tokenize(self.train_df.iloc[index]['review'] if self.train else self.test_df.iloc[index]['review'])

        return label, text

    def __len__(self):
        if self.train:
            return self.train_df.shape[0]
        else:
            return self.test_df.shape[0]
        
    def tokenize(self, text):
        text = [i.strip() for i in text.split()]
        text = [i.lower() for i in text]
        text = [re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", i) for i in text]
        return text


def get_dataloader(path, batch_size=8, train=True):
    imdb_dataset = ImdbDataset(path, clean=True, shuffle=True, train=train)
    dataloader = DataLoader(imdb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader


def collate_fn(batch):
    '''
    param batch: ([labels, tokens]， [labels, tokens], 一个getitem的结果...)
    '''
    #batch是list, 其中一个一个元组，每个元组是dataset中__getitem__的结果
    text, label = list(zip(*batch))
    return text, label


if __name__ == '__main__':
    path = r"C:\Users\45323\OneDrive\桌面\新python文件夹\pytorch\IMDB Dataset.csv"
    for i, (input, target) in enumerate(get_dataloader(path)):
        print("label: {}, text: {}".format(input, target))
        print("*"*100)
        break



#NOTE: 以下会报错
# 解决方法1：先转化成数字
# 解决方法2：写个collate_fn

# if __name__ == '__main__':
#     for i, (input, target) in enumerate(get_dataloader()):
#         print("label: {}, text: {}".format(input, target))
#         print("*"*100)
#         break