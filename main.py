from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
import requests

app = FastAPI()

# Hugging Face Whisper API (external inference)
def transcribe_audio_hf(file_path):
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not api_token:
        raise Exception("Hugging Face API token not found in environment variables.")

    API_URL = "https://api-inference.huggingface.co/models/openai/whisper"
    headers = {"Authorization": f"Bearer {api_token}"}

    with open(file_path, "rb") as f:
        response = requests.post(API_URL, headers=headers, data=f)
        response.raise_for_status()
        result = response.json()
        return result.get("text", "")

# Lightweight keyword-based stress detection
def detect_stress(text):
    stress_keywords = ["anxious", "worried", "tired", "pressure", "panic", "overwhelmed"]
    for word in text.lower().split():
        if word in stress_keywords:
            return "stressed"
    return "not_stressed"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        transcription = transcribe_audio_hf(file_location)
        prediction = detect_stress(transcription)
        os.remove(file_location)
        return JSONResponse(content={"transcription": transcription, "prediction": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
