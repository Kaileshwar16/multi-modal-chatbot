from model.text_encoder import TextEncoder, SimpleTokenizer
from model.image_encoder import ImageEncoder
from model.retriever import Retriever
from model.generator import Seq2SeqGenerator
import torch.nn as nn

class RAGModel:
    def __init__(self, embed_size=128, hidden_size=256, device="cpu"):
        self.device = device
        self.tokenizer = SimpleTokenizer()
        self.dataset = []

        self.text_encoder = None  # initialize later after tokenizer is fit
        self.image_encoder = ImageEncoder(device=device)
        self.generator = None
        self.retriever = None

    def index_documents(self, dataset):
        """
        Fit tokenizer, initialize components, and store image embeddings in retriever.
        dataset: List of dicts with 'image', 'question', 'answer'
        """
        self.dataset = dataset

        # Fit tokenizer
        all_texts = [item['question'] for item in dataset]
        self.tokenizer.fit(all_texts)

        vocab_size = self.tokenizer.vocab_size()

        # Initialize modules after vocab is known
        self.text_encoder = TextEncoder(vocab_size, embed_dim=128, tokenizer=self.tokenizer)
        self.generator = Seq2SeqGenerator(vocab_size, embed_size=128, hidden_size=256, device=self.device)
        self.retriever = Retriever(self.text_encoder, self.image_encoder, self.device)

        # Encode and index images
        for item in dataset:
            image_path = item['image_path'] if 'image_path' in item else item['image']
            embedding = self.image_encoder.encode(image_path)
            self.retriever.add(item, embedding)

    def answer_question(self, question, image_path):
        """
        Uses image and question to retrieve relevant info and generate answer.
        """
        query_embedding = self.image_encoder.encode(image_path)
        retrieved = self.retriever.retrieve(query_embedding)

        context = retrieved['answer']  # using answer as proxy context for now
        return self.generator.generate_answer(question, context)
