import gradio as gr
from model.rag_model import MultimodalRAG

rag_model = MultimodalRAG(device="cpu")  # ✅ removed vocab_size
rag_model.load_data("data/train.json", "data/images")

def predict(image, question):
    query_path = "data/images/query.png"
    image.save(query_path)
    answer = rag_model.answer_question(question, query_path)
    return answer

iface = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Question")],
    outputs=gr.Textbox(label="Answer"),
    title="Multimodal RAG Bot"
)

iface.launch()
