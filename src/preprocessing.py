# src/preprocessing.py ← FINAL 100% WORKING
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Convert uploaded image bytes → [1, 3, 224, 224] tensor for ResNet18
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension → [1, 3, 224, 224]