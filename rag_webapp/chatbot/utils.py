import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

# Prevent TensorFlow warnings
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Load models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def build_vector_store(text, chunk_size=200):
    # Sanity check
    if not text:
        raise ValueError("Input text is empty. PDF might be blank or unreadable.")

    # Split into chunks
    chunks = [text[i:i + chunk_size].strip() for i in range(0, len(text), chunk_size)]
    chunks = [chunk for chunk in chunks if chunk]  # Filter out empty chunks

    if not chunks:
        raise ValueError("No valid text chunks generated from the input text.")

    # Generate embeddings
    embeddings = embedder.encode(chunks)

    # Validate embeddings
    if not hasattr(embeddings, "shape") or len(embeddings.shape) != 2:
        raise ValueError("Embedding generation failed. Output shape is invalid.")

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, chunks

def retrieve_relevant_chunk(question, index, chunks, k=1):
    question_vec = embedder.encode([question])
    D, I = index.search(np.array(question_vec), k)
    return " ".join([chunks[i] for i in I[0]])

def answer_question(question, context):
    return qa_pipeline(question=question, context=context)["answer"]
