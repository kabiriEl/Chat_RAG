import google.generativeai as genai
import os

# Remplace par ta propre clé
GEMINI_API_KEY = "AIzaSyDDcS3AzRjEp4w1gEFVeRbzOR_nC3SYfYI"
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

def reformulate_plantnet_response(plantnet_json: dict) -> str:
    prompt = f"""
    Tu es un assistant intelligent pour les agriculteurs.
    Voici un résultat brut de l'API Pl@ntNet. Reformule-le en langage simple, compréhensible par un agriculteur, en expliquant quelle plante est identifiée, avec quel niveau de confiance, et ses usages possibles si connus :

    
    {plantnet_json}
    

    Réponds uniquement en français, de façon concise, claire et utile.
    """

    response = model.generate_content(prompt)
    return response.text
