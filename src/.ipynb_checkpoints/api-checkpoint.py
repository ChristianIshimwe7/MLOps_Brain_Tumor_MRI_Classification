from fastapi import FastAPI, File, UploadFile
from src.model import load_model
from src.preprocessing import preprocess_image
import torch

app = FastAPI()
model = load_model()

@app.get("/health")
def health(): return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    bytes_ = await file.read()
    tensor = preprocess_image(bytes_)
    with torch.no_grad():
        prob = torch.softmax(model(tensor), dim=1)[0]
        pred = int(prob.argmax())
        conf = float(prob.max())
    return {
        "prediction": "Tumor Present" if pred == 1 else "No Tumor",
        "confidence": round(conf, 3)
    }