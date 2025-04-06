import re

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, texts):
        for text in texts:
            for word in self.tokenize(text):
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def encode(self, text):
        return [self.word2idx.get(word, 1) for word in self.tokenize(text)]

    def vocab_size(self):
        return len(self.word2idx)
