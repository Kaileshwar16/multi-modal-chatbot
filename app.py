import gradio as gr
from model.rag_model import MultimodalRAG

# Initialize model
rag_model = MultimodalRAG(device="cpu")
sample_texts = [
    "question: What is the total amount? context: The total amount due is ₹1500.",
    "question: Who is the customer? context: The customer is ABC Pvt Ltd.",
    "question: What is the invoice number? context: Invoice number is INV-2025-001.",
    "question: What is the due date? context: Due date is 10th April 2025."
]
rag_model.load_model(sample_texts=sample_texts)

def predict(question, file):
    file_path = file.name
    answer = rag_model.answer_question(question, file_path)
    return answer

# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Enter your question"),
        gr.File(label="Upload document (PDF or image)")
    ],
    outputs="text",
    title="Multimodal Document QA",
    description="Upload a document (PDF/image) and ask a question about it."
)

interface.launch()
