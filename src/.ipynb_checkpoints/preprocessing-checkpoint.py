import io
from PIL import Image
import torch
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(file_bytes))
    return preprocess(img).unsqueeze(0)