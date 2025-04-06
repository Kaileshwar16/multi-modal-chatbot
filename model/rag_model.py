import torch
import torch.nn as nn
import pytesseract
from PIL import Image
import os
from pdf2image import convert_from_path
from model.text_encoder import TextEncoder
from model.tokenizer import SimpleTokenizer
from model.image_encoder import ImageEncoder
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
        self.generator = None

    # ✅ Added `sample_texts` parameter to fit tokenizer properly
    def load_model(self, sample_texts=None):
        if sample_texts is None:
            # Fallback dummy text (in case nothing is passed)
            sample_texts = ["This is a sample question for tokenizer."]

        # ✅ Train tokenizer on actual sample texts
        self.tokenizer.fit(sample_texts)

        vocab_size = self.tokenizer.vocab_size()
        self.text_encoder = TextEncoder(vocab_size, embed_dim=self.embed_size).to(self.device)
        self.text_encoder.tokenizer = self.tokenizer

        self.generator = Seq2SeqGenerator(vocab_size, self.embed_size, self.hidden_size, self.device).to(self.device)
        self.generator.tokenizer = self.tokenizer

    def extract_text_from_doc(self, file_path):
        text = ""
        if file_path.lower().endswith(".pdf"):
            pages = convert_from_path(file_path)
            for page in pages:
                text += pytesseract.image_to_string(page)
        else:
            image = Image.open(file_path).convert("RGB")
            text = pytesseract.image_to_string(image)
        return text.strip()

    def answer_question(self, question, file_path):        
        context = self.extract_text_from_doc(file_path)
        if not context:
            return "Could not extract text from the document."

        self.tokenizer.fit([context, question])

        input_text = f"question: {question} context: {context}"
        tokenized = self.tokenizer.encode(input_text)
        print("Encoded Input:", tokenized)  # 🧪 Debug
        input_tensor = torch.tensor(tokenized).unsqueeze(0).to(self.device)

        output_ids = self.generator.generate(input_tensor)
        print("Decoded token IDs:", output_ids[0])  # 🧪 Debug
        answer = self.tokenizer.decode(output_ids[0])
        return answer
