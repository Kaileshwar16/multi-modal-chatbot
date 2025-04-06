import torch
import torch.nn as nn
import os
import json
from model.text_encoder import TextEncoder
from model.tokenizer import SimpleTokenizer
from model.image_encoder import ImageEncoder
from model.retriever import Retriever
from model.generator import Seq2SeqGenerator

class MultimodalRAG(nn.Module):
    def __init__(self, embed_size=128, hidden_size=256, device="cpu"):
        super(MultimodalRAG, self).__init__()
        self.device = device
        self.embed_size = embed_size   
        self.hidden_size = hidden_size
        self.tokenizer = SimpleTokenizer()
        self.text_encoder = None
        self.image_encoder = ImageEncoder()
        self.retriever = None
        self.generator = None
        self.dataset = []

    def load_data(self, json_path, image_folder):
        with open(json_path, "r") as f:
            self.dataset = json.load(f)

        for item in self.dataset:
            img_name = os.path.splitext(item["image"])[0] + ".png"
            item["image_path"] = os.path.join(image_folder, img_name)

        all_texts = [item["question"] for item in self.dataset]
        self.tokenizer.fit(all_texts)

        vocab_size = self.tokenizer.vocab_size()
        self.text_encoder = TextEncoder(vocab_size, embed_dim=self.embed_size).to(self.device)
        self.text_encoder.tokenizer = self.tokenizer

        self.generator = Seq2SeqGenerator(vocab_size, self.embed_size, self.hidden_size, self.device).to(self.device)
        self.retriever = Retriever(self.text_encoder, self.image_encoder, self.device)
        self.retriever.index(self.dataset)

    def answer_question(self, question, image_path):
        best_match = self.retriever.retrieve(question, [image_path])
        for item in self.dataset:
            if item["image_path"] == best_match:
                return item["answer"]
        return "No answer found."
