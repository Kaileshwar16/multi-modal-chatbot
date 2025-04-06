class MultimodalRAG(nn.Module):
    def __init__(self, embed_size=128, hidden_size=256, device="cpu"):
        super(MultimodalRAG, self).__init__()
        self.device = device
        self.tokenizer = SimpleTokenizer()

        self.text_encoder = None  # Delay init until tokenizer is fitted
        self.image_encoder = ImageEncoder()
        self.retriever = None  # Delay init
        self.generator = None  # Delay init
        self.dataset = []

    def load_data(self, json_path, image_folder):
        with open(json_path, "r") as f:
            self.dataset = json.load(f)

        # Normalize image paths
        for item in self.dataset:
            img_name = os.path.splitext(item["image"])[0] + ".png"
            item["image_path"] = os.path.join(image_folder, img_name)

        # Fit tokenizer
        all_texts = [item["question"] for item in self.dataset]
        self.tokenizer.fit(all_texts)

        # Now that tokenizer is fitted, get vocab size
        vocab_size = self.tokenizer.vocab_size()

        # Initialize encoders and retriever with tokenizer
        self.text_encoder = TextEncoder(vocab_size, embed_dim=128, tokenizer=self.tokenizer).to(self.device)
        self.generator = Seq2SeqGenerator(vocab_size, 128, 256, self.device).to(self.device)
        self.retriever = Retriever(self.text_encoder, self.image_encoder, self.device)

        # Index the dataset
        self.retriever.index(self.dataset)

    def answer_question(self, question, image_path):
        best_match = self.retriever.retrieve(question, [image_path])
        for item in self.dataset:
            if item["image_path"] == best_match:
                return item["answer"]
        return "No answer found."

    def forward(self, question, doc_texts, image_tensors, target_seq=None, teacher_forcing_ratio=0.5):
        question_vec = self.text_encoder(question)
        doc_vecs = [self.text_encoder(doc) for doc in doc_texts]
        img_vecs = [self.image_encoder(img.unsqueeze(0)) for img in image_tensors]

        all_chunks = doc_vecs + img_vecs
        retrieved_vecs = self.retriever.retrieve(question_vec, all_chunks, k=1)
        context_vec = torch.cat([question_vec] + retrieved_vecs, dim=1)
        output = self.generator(context_vec, target_seq, teacher_forcing_ratio)
        return output
