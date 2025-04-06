import json
import os
from model.rag_model import MultimodalRAG
from utils import load_image
import torch

def load_dataset(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset("data/train.json")
    print(f"Loaded {len(dataset)} samples.")

    # Initialize model
    rag = MultimodalRAG(device=device)
    rag.load_data(dataset)

    # Run evaluation
    print("\nRunning model on dataset:")
    for item in dataset:
        question = item["question"]
        expected = item["answer"]
        predicted = rag.answer_question(question)
        print(f"\nQ: {question}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")

if __name__ == "__main__":
    main()
