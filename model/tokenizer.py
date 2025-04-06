class SimpleTokenizer:
    def __init__(self):
        # Special tokens
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.word2idx = self.vocab.copy()
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
    
    def fit(self, texts):
        """Build vocabulary from a list of input texts."""
        idx = len(self.vocab)
        for text in texts:
            for word in text.strip().lower().split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

    def encode(self, text):
        """Convert text to a list of token IDs."""
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in text.strip().lower().split()]

    def decode(self, token_ids):
        """Convert token IDs back to a text string."""
        words = [self.idx2word.get(idx, "<UNK>") for idx in token_ids]
        print("Decoded token IDs:", token_ids)  # Debug output
        return ' '.join(words)

    def vocab_size(self):
        """Return the size of the vocabulary."""
        return len(self.vocab)
