import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os

st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")
st.title("Brain Tumor MRI Classifier")
st.markdown("---")

# Model path
model_path = "models/brain_tumor_model.pth"

if not os.path.exists(model_path):
    st.error(f"Model not found! Make sure '{model_path}' is in your GitHub repo!")
    st.stop()

# Load model
@st.cache_resource
def load_model():
    with st.spinner("Loading your trained model... (first time ~20 seconds)"):
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
        state_dict = torch.load(model_path, map_location="cpu")
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
    return model

model = load_model()
st.success("Model loaded successfully!")

# Preprocess
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Upload
uploaded_file = st.file_uploader("Upload Brain MRI", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)
    
    if st.button("Analyze Image", type="primary"):
        with st.spinner("Analyzing..."):
            tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)[0]
                confidence, predicted_class = torch.max(probs, 0)
                
                conf = confidence.item() * 100
                
                # FINAL FIX: Your model uses class 1 = Tumor, class 0 = No Tumor
                if predicted_class.item() == 1:
                    result = "Tumor Present"
                else:
                    result = "No Tumor"
        
        st.markdown("---")
        if result == "Tumor Present":
            st.error(f"**{result}**")
            st.warning(f"Confidence: {conf:.2f}%")
            st.info("Please consult a medical professional.")
        else:
            st.success(f"**{result}**")
            st.info(f"Confidence: {conf:.2f}%")

st.caption("By Christian Ishimwe – Rwanda's MLOps Legend")