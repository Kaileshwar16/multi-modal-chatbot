import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
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

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, 512)
        self.tokenizer = tokenize

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        pooled = embedded.mean(dim=1)         # simple average pooling
        out = self.fc(pooled)
        return out

def collate_fn(tokenizer, texts):
    encoded = [torch.tensor(tokenizer.encode(text)) for text in texts]
    padded = pad_sequence(encoded, batch_first=True, padding_value=0)
    return padded
