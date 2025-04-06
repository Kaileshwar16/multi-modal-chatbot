import torch.nn as nn
import torch

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, 512)
        self.tokenizer = None

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        pooled = embedded.mean(dim=1)
        return self.fc(pooled)
