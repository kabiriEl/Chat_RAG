from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uvicorn

from rag_utils import load_vector_store, get_answer
from image_predictor import predict_plant_disease
from plantnet_api import router as plantnet_router  # 👈 importer le router
from fastapi import UploadFile, File

app = FastAPI()

# Initialiser le retriever
vector_store, retriever = load_vector_store()

# Schéma de question texte
class Question(BaseModel):
    query: str

# Endpoint pour chatbot texte
@app.post("/ask")
async def ask_question(q: Question):
    response = get_answer(q.query, retriever)
    return JSONResponse(content={"answer": response}, media_type="application/json; charset=utf-8")

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    return await predict_plant_disease(file)

# 🚀 Inclure le router Pl@ntNet
app.include_router(plantnet_router)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
