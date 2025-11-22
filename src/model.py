# src/model.py  ← FIXED VERSION (CPU ONLY)
import torch
import torch.nn as nn
from torchvision import models
import os

def load_model(path="models/brain_tumor_model.pth"):
    # Force CPU – fixes DLL error on Windows
    device = torch.device("cpu")
    torch.set_default_device(device)
    
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Load your trained weights
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print("Model loaded on CPU (DLL error fixed!)")
    else:
        print("Warning: Model file not found – using random weights")
    
    model.eval()
    return model