import torch.nn as nn
from torchvision import models

def load_model(path="models/brain_tumor_resnet.pth"):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model