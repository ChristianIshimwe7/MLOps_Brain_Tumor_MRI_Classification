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

# Check model exists
model_path = "models/brain_tumor_model.pth"
if not os.path.exists(model_path):
    st.error(f"Model not found! Make sure {model_path} is in your GitHub repo.")
    st.stop()

@st.cache_resource
def load_model():
    with st.spinner("Loading your trained model... (first time ~20s)"):
        # Load the EXACT same architecture you used during training
        # Most brain tumor projects use ResNet18 → this works 99% of the time
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: tumor / no tumor
        
        # Load your trained weights
        state_dict = torch.load(model_path, map_location="cpu")
        
        # Fix if model was saved with DataParallel
        if "module" in list(state_dict.keys())[0]:
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.'
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
    return model

model = load_model()
st.success("Model loaded successfully!")

# Preprocessing
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
    
    if st.button("Analyze", type="primary"):
        with st.spinner("Analyzing..."):
            tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)
                
                result = "Tumor Present" if pred.item() == 1 else "No Tumor"
                conf = confidence.item() * 100
                
        st.markdown("---")
        if "Tumor" in result:
            st.error(f"**{result}**")
            st.warning(f"Confidence: {conf:.2f}%")
            st.info("Please consult a doctor immediately.")
        else:
            st.success(f"**{result}**")
            st.info(f"Confidence: {conf:.2f}%")