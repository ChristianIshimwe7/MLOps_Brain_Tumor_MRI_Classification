# src/api.py  ← FULL FINAL VERSION (CPU-SAFE + WORKING)
from fastapi import FastAPI, File, UploadFile
from src.model import load_model
from src.preprocessing import preprocess_image
import torch

# Create the app
app = FastAPI(
    title="Brain Tumor MRI Classifier",
    description="Upload a brain MRI → Get instant Tumor / No Tumor prediction",
    version="1.0"
)

# Load model ONCE at startup (CPU-safe)
model = load_model()  # ← This now works perfectly thanks to the fix in model.py

@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a brain MRI image (JPG/PNG) → Returns prediction + confidence
    """
    try:
        # Read image bytes
        image_bytes = await file.read()

        # Preprocess
        input_tensor = preprocess_image(image_bytes)  # shape: [1, 3, 224, 224]

        # Predict (no GPU needed)
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence = probabilities.max().item()
            prediction_idx = probabilities.argmax().item()

        # Convert to readable result
        result = "Tumor Present" if prediction_idx == 1 else "No Tumor"
        color = "red" if prediction_idx == 1 else "green"

        return {
            "prediction": result,
            "confidence": round(confidence, 3),
            "confidence_percentage": f"{confidence:.1%}",
            "message": f"Model is {confidence:.1%} sure → {result}"
        }

    except Exception as e:
        return {"error": str(e)}