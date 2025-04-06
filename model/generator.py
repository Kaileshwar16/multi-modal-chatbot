import torch.nn as nn
import torch

class Seq2SeqGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, device="cpu"):
        super(Seq2SeqGenerator, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.tokenizer = None  # will be attached later

    def forward(self, input_tensor):
        embedded = self.embedding(input_tensor)
        _, (hidden, cell) = self.encoder(embedded)
        decoder_input = torch.zeros((input_tensor.size(0), 1), dtype=torch.long).to(self.device)

        outputs = []
        for _ in range(20):  # max output length
            embedded_dec = self.embedding(decoder_input)
            output, (hidden, cell) = self.decoder(embedded_dec, (hidden, cell))
            token_logits = self.out(output.squeeze(1))
            predicted_token = token_logits.argmax(1, keepdim=True)
            outputs.append(predicted_token)
            decoder_input = predicted_token

        return torch.cat(outputs, dim=1)

    def generate(self, input_tensor):
        output_ids = self.forward(input_tensor)
        output_text = self.tokenizer.decode(output_ids[0].tolist())
        return output_text

