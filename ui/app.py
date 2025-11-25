import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")

# Header
st.title("Brain Tumor MRI Classifier")
st.markdown("---")

# Check if model exists
if not os.path.exists("models/brain_tumor_model.pth"):
    st.error("Model file not found! Make sure `models/brain_tumor_model.pth` is uploaded to your GitHub repo.")
    st.stop()

# Load model (cached)
@st.cache_resource
def load_model():
    with st.spinner("Loading AI model (first time may take ~20 seconds)..."):
        # Use ResNet50 backbone (same as your training)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)  # 2 classes: tumor / no tumor
        
        # Load your trained weights
        model.load_state_dict(torch.load("models/brain_tumor_model.pth", map_location="cpu"))
        model.eval()
    return model

model = load_model()
st.success("Model loaded successfully!")

# Image preprocessing
def preprocess_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img)
    return tensor.unsqueeze(0)  # Add batch dimension

# File uploader
uploaded_file = st.file_uploader("Upload Brain MRI Scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)
    
    if st.button("Analyze Image", type="primary"):
        with st.spinner("AI is analyzing the MRI..."):
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                conf = confidence.item() * 100
                result = "Tumor Present" if predicted_class.item() == 1 else "No Tumor"
        
        st.markdown("---")
        if "Tumor" in result:
            st.error(f"**{result}**")
            st.warning(f"Confidence: {conf:.2f}%")
            st.info("Please consult a medical professional immediately.")
        else:
            st.success(f"**{result}**")
            st.info(f"Confidence: {conf:.2f}%")