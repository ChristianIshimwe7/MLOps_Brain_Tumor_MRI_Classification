import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# Load model directly (no FastAPI needed)
@st.cache_resource
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes
    model.load_state_dict(torch.load("model/brain_tumor_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Preprocess image
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Title
st.title("Brain Tumor MRI Classifier")

uploaded_file = st.file_uploader("Upload MRI", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)
    
    with st.spinner("Analyzing..."):
        tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            result = "Tumor Present" if predicted.item() == 1 else "No Tumor"
            conf = confidence.item() * 100
            
            if result == "Tumor Present":
                st.error(f"**{result}**")
                st.warning(f"Confidence: {conf:.1f}%")
            else:
                st.success(f"**{result}**")
                st.info(f"Confidence: {conf:.1f}%")