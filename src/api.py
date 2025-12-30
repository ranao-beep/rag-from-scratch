from fastapi import FastAPI
from src.rag import rag_answer

app = FastAPI()


@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.get("/search")
def search(query: str):
    print("Received query:", query)
    answer = rag_answer(query)
    print("Generated answer")
    return answer

