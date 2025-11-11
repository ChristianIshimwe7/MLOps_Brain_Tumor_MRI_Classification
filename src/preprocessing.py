import io
from PIL import Image
import torch
from torchvision import transforms

normalize = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(file_bytes))
    return normalize(img).unsqueeze(0)  # (1, 1, 28, 28)
