from fastapi import FastAPI, File, UploadFile
from src.model import load_model
from src.preprocessing import preprocess_image
import torch

app = FastAPI(title="Brain Tumor Classifier")
model = load_model()

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    tensor = preprocess_image(image_bytes)
    with torch.no_grad():
        prob = torch.softmax(model(tensor), dim=1)[0]
        conf = prob.max().item()
        pred = "Tumor Present" if prob.argmax() == 1 else "No Tumor"
    return {"prediction": pred, "confidence": round(conf, 4)}