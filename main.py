import pickle
from dataset import get_dataloader
from word_sequence  import Word2Sequence
from tqdm import tqdm



if __name__ == '__main__':
    w2s = Word2Sequence()
    path = r"C:\Users\45323\OneDrive\桌面\新python文件夹\pytorch\IMDB Dataset.csv"
    for i, (target, input) in tqdm(enumerate(get_dataloader(path, batch_size=2))):
        for text in input:
            w2s.fit(text)
    w2s.build_vocab(min=10, max_features=7000) # 会多两个：total vocab size: 7002, 因为有self.dict的两个额外参数
    pickle.dump(w2s, open("models/w2s.pkl", "wb")) # wb because we are writing it/ creating a new one
    print(len(w2s))