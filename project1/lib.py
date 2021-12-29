import pickle

from torch.nn.functional import embedding

path = r"C:\Users\45323\OneDrive\桌面\新python文件夹\pytorch\project1\models\w2s.pkl"
ws = pickle.load(open(path, "rb")) # rb because we are loading it
print(ws)

max_len = 20
embed_dim = 100