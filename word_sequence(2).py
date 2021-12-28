"""
实现的是: 构建词典，实现方法吧句子转化成数字序列何其翻转
"""

class Word2Sequence:
    UNK_TAG = "UNK" # 未知符号
    PAD_TAG = "PAD" # padding符号

    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD,
        }
        
        self.count = {}

    def fit(self, sentence):
        """吧单个句子保存到dict当中
        param: sentence: [word1, word2, word3...]
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min=5, max=None, max_features=None):
        """
        生成词典
        param min:          最小出现的次数
        param max:          最大的次数
        param max_features: 一共保留多少的词语
        returns:            
        """
        # 删除count中词频小于min的word
        if min is not None:
            self.count = {word: value for word,value in self.count.items() if value > min}
        # 删除次数大于max的值
        if max is not None:
            self.count = {word: value for word,value in self.count.items if value < max}
        # 限制保留的词语数
        if max_features is not None:
            temp = sorted(self.count.items(), key=lambda x:x[-1], reverse=True)[:max_features]
            self.count = dict(temp)

        for word in self.count:
            self.dict[word] = len(self.dict)
        
        # 得到一个翻转的dict字典
        self.reverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
    
    def transform(self, sentence, max_len=None):
        """
        把句子转化成序列
        param sentence: [word1, word2...]
        param max_len: int, 对句子精选填充或者裁剪
        """
        if max_len is not None: # do padding here 填充
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG] * (max_len-len(sentence))
            if max_len < len(sentence):
                sentence = sentence[:max_len] #裁剪

        return [self.dict.get(word, self.UNK) for word in sentence]
    
    def inverse_transform(self, indices):
        """
        靶序列转化为句子
        param indices: [1, 2, 3, 4, 5...]
        """
        return [self.reverse_dict.get(idx) for idx in indices]


if __name__ == '__main__':
    w2s = Word2Sequence()
    w2s.fit(['I', 'am', 'your', 'father'])
    w2s.build_vocab(min=0)
    ret = w2s.transform(['your', '奶奶', '的', '给', '我', '玩阴', '的', '是', '吧', '？'], max_len=10)
    print(w2s.dict)
    print(ret)