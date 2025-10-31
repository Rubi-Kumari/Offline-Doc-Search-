from transformers import pipeline
import torch

# Load local QA model (no API key required)
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad",
    device=0 if torch.cuda.is_available() else -1
)

def answer_question(question, context):
    try:
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    except Exception as e:
        return "⚠️ Sorry, couldn't extract a clear answer."