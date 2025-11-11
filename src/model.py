import torch.nn as nn
from torchvision import models

def load_model(path="models/resnet_finetuned.pth", device="cpu"):
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)
