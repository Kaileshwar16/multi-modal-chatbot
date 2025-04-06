import torch.nn.functional as F
import torch
from model.image_encoder import load_image

class Retriever:
    def __init__(self, text_encoder, image_encoder, device="cpu"):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.device = device
        self.dataset = []

    def index(self, dataset):
        self.dataset = dataset

    def retrieve(self, question, image_paths):
        token_ids = torch.tensor([self.text_encoder.tokenizer.encode(question)]).to(self.device)
        q_embedding = self.text_encoder(token_ids)

        best_score = -1
        best_path = None
        for img_path in image_paths:
            image_tensor = load_image(img_path).unsqueeze(0).to(self.device)
            img_embedding = self.image_encoder(image_tensor)
            score = F.cosine_similarity(q_embedding, img_embedding).item()
            if score > best_score:
                best_score = score
                best_path = img_path

        return best_path
