import pickle
import torch
from torch.nn.functional import embedding

path = r"C:\Users\45323\OneDrive\桌面\新python文件夹\pytorch\project1\models\w2s.pkl"
ws = pickle.load(open(path, "rb")) # rb because we are loading it


max_len = 200
embed_dim = 100
hidden_size = 128
num_layers = 2
bidirectional = True
dropout = 0.5
batch_size = 512
test_batch_size = 1000
# GPU配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    print(ws)