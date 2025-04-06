import torch.nn as nn
import torch

class Seq2SeqGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, device="cpu"):
        super(Seq2SeqGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.device = device

    def forward(self, context_vec, target_seq=None, teacher_forcing_ratio=0.5):
        return torch.argmax(context_vec, dim=-1)

