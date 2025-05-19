from django.shortcuts import render
from .utils import extract_text_from_pdf, build_vector_store, retrieve_relevant_chunk, answer_question
import os

os.environ["TRANSFORMERS_NO_TF"] = "1"


def home(request):
    answer = None
    error = None

    if request.method == "POST" and request.FILES.get("document"):
        uploaded_file = request.FILES["document"]
        question = request.POST.get("question")

        # Save uploaded file temporarily
        temp_path = "temp.pdf"
        with open(temp_path, "wb+") as dest:
            for chunk in uploaded_file.chunks():
                dest.write(chunk)

        try:
            text = extract_text_from_pdf(temp_path)

            if not text.strip():
                error = "The uploaded PDF contains no readable text. It may be image-based or blank."

            else:
                index, chunks = build_vector_store(text)
                context = retrieve_relevant_chunk(question, index, chunks)
                answer = answer_question(question, context)

        except Exception as e:
            error = f"An error occurred: {str(e)}"

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return render(request, "chatbot/index.html", {
        "answer": answer,
        "error": error
    })
