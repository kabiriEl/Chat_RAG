from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import requests
import shutil
import os
from tempfile import NamedTemporaryFile
from gemini_utils2 import reformulate_plantnet_response  # ⬅️ Gemini reformulation

router = APIRouter()

PLANTNET_API_URL = "https://my-api.plantnet.org/v2/identify/all"
PLANTNET_API_KEY = "2b10t5yfGyF2hY1ouVQgDB9E"

@router.post("/identify-plant")
async def identify_plant(file: UploadFile = File(...)):
    try:
        # 1. Sauvegarde temporaire du fichier
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # 2. Préparer la requête Pl@ntNet
        with open(temp_file_path, "rb") as image_file:
            files = {'images': (file.filename, image_file, file.content_type)}
            data = {'organs': 'leaf'}
            response = requests.post(
                f"{PLANTNET_API_URL}?api-key={PLANTNET_API_KEY}",
                files=files,
                data=data,
            )

        os.remove(temp_file_path)  # Nettoyage

        if response.status_code != 200:
            return JSONResponse(
                status_code=response.status_code,
                content={"error": "Pl@ntNet API failed", "details": response.text},
            )

        plantnet_data = response.json()

        # 3. Reformuler avec Gemini 1.5 Flash
        reformulated = reformulate_plantnet_response(plantnet_data)

        # 4. Retourner les deux réponses
        return  reformulated
        

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
