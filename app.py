from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load the Hugging Face pipeline for text generation using your model
chatbot = pipeline("text-generation", model="w3dhamma-ai")

@app.get("/")
def read_root():
    return {"message": "Welcome to the W3Dhamma AI Chatbot!"}

@app.post("/ask")
def ask_question(question: str):
    # Get the response from the model
    response = chatbot(question, max_length=1000, num_return_sequences=1)
    return {"response": response[0]['generated_text']}
