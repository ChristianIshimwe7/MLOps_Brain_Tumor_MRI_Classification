# src/model.py ← FINAL CPU-SAFE VERSION
import torch
import torch.nn as nn
from torchvision import models
import os

def load_model(path="models/brain_tumor_model.pth"):
    device = torch.device("cpu")  # Force CPU → no DLL errors
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print("Model loaded successfully on CPU!")
    else:
        print("Model file not found → using random weights (for demo)")
    
    model.eval()
    return model.to(device)