# utils.py

import json
import re

def tokenize(text, vocab, max_len=20):
    tokens = re.findall(r"\b\w+\b", text.lower())
    token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    token_ids = token_ids[:max_len]
    # Pad to max_len
    while len(token_ids) < max_len:
        token_ids.append(vocab["<pad>"])
    return token_ids

def detokenize(token_ids, inv_vocab):
    tokens = [inv_vocab.get(idx, "<unk>") for idx in token_ids]
    text = " ".join(tokens)
    return text.replace(" <pad>", "").strip()

def build_vocab(json_path, min_freq=1):
    word_freq = {}
    with open(json_path) as f:
        data = json.load(f)
        for item in data:
            for field in ["question", "text", "answer"]:
                tokens = re.findall(r"\b\w+\b", item[field].lower())
                for token in tokens:
                    word_freq[token] = word_freq.get(token, 0) + 1

    vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
    idx = 4
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab
