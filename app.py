import gradio as gr
from model.rag_model import MultimodalRAG


rag_model = MultimodalRAG(device="cpu")
sample_texts = [
    "question: What is the total amount? context: The total amount due is ₹1500.",
    "question: Who is the customer? context: The customer is ABC Pvt Ltd.",
    "question: What is the invoice number? context: Invoice number is INV-2025-001.",
    "question: What is the due date? context: Due date is 10th April 2025.",
    "question: What is the amount? context: The amount is ₹2000.",
    "question: When should the payment be made? context: The payment is due by 15th April.",
    "question: Who issued the invoice? context: Issued by XYZ Corporation."
    "question: What is the invoice number? context: Iwoce numeeR) INV-2025-001"
    "question: Who is the customer? context: AaumG mores Conpry e"
    "question: What is the due date? context: Due date: 10 April 2025"
    "question: What is the total amount? context: Total ₹1500 only"
    "question: What is the payment method? context: Powa Mathod: UPI"

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
