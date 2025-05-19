from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag_utils import load_vector_store , get_answer
import uvicorn

app = FastAPI()

# Charger l'index FAISS + le modèle d'embedding au démarrage
vector_store, retriever = load_vector_store()

class Question(BaseModel):
    query: str 

@app.post("/ask")
async def ask_question(q: Question):
    response = get_answer(q.query, retriever)
    return {"answer": response}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
