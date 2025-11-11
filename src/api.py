from fastapi import FastAPI, File, UploadFile
from src.model import load_model
from src.preprocessing import preprocess_image
import torch

app = FastAPI()
device = torch.device('cpu')
model = load_model(device=device)

@app.get("/health")
def health(): return {"status": "up"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    bytes_ = await file.read()
    tensor = preprocess_image(bytes_).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(tensor), dim=1)
        pred = int(prob.argmax(dim=1).item())
    return {
        "prediction": "horizontal" if pred == 0 else "vertical",
        "confidence": float(prob.max().item())
    }
