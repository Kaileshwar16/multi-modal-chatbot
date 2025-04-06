🧠 Multimodal RAG Chatbot + Invoice Extractor

This project is a Django-based application that allows users to upload a PDF document (e.g. reports, invoices) and ask questions about its content. It uses a Retrieval-Augmented Generation (RAG) pipeline to find relevant text chunks and generate answers using a pre-trained Question-Answering model.

It also includes a simple invoice parser to handle invoice-like documents for data extraction tasks.

🚀 Features

    Upload any PDF

    ❓ Ask natural language questions about the content

    🔍 Semantic search with Sentence Transformers + FAISS

    🤖 Get answers from a QA model (DistilBERT)

    🧾 Basic invoice parser to process invoice documents and extract fields


📦 Tech Stack

    Backend: Django

    Semantic Embeddings: SentenceTransformers

    Similarity Search: FAISS

    Question Answering: Hugging Face Transformers (DistilBERT)

    PDF Handling: PyMuPDF (fitz)

    Invoice Parser: Simple Python logic (can be extended)



# 1. Clone the repo
git clone https://github.com/Kaileshwar16/multi-modal-chatbot.git
cd multimodal_rag
cd rag_webapp

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Apply migrations
python manage.py migrate

# 5. Run the development server
python manage.py runserver
