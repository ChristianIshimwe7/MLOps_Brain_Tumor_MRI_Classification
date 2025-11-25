import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os

# ========================
#  CONFIGURATION
# ========================
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")
st.title("Brain Tumor MRI Classifier")
st.markdown("---")

model_path = "models/brain_tumor_model.pth"

if not os.path.exists(model_path):
    st.error(f"Model not found! Make sure the file '{model_path}' is in your GitHub repo.")
    st.stop()

# ========================
#  LOAD MODEL
# ========================
@st.cache_resource
def load_model():
    with st.spinner("Loading model... (first time takes ~20 seconds)"):
        # Most brain tumor projects use ResNet18
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes
        
        # Load your trained weights
        state_dict = torch.load(model_path, map_location="cpu")
        
        # Remove 'module.' prefix if model was saved with DataParallel
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
    return model

model = load_model()
st.success("Model loaded successfully!")

# ========================
#  PREPROCESS IMAGE
# ========================
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)  # add batch dimension

# ========================
#  UPLOAD & PREDICT
# ========================
uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)
    
    if st.button("Analyze Image", type="primary"):
        with st.spinner("Analyzing the MRI..."):
            tensor = preprocess_image(image)
            
            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                confidence, predicted_class = torch.max(probabilities, 0)
                
                conf_percent = confidence.item() * 100
                
                # THIS LINE FIXES YOUR PROBLEM:
                # Many models have class 0 = Tumor, class 1 = No Tumor
                if predicted_class.item() == 0:
                    result = "Tumor Present"
                else:
                    result = "No Tumor"
        
        st.markdown("---")
        if result == "Tumor Present":
            st.error(f"**{result}**")
            st.warning(f"Confidence: {conf_percent:.2f}%")
            st.info("Please consult a doctor immediately.")
        else:
            st.success(f"**{result}**")
            st.info(f"Confidence: {conf_percent:.2f}%")

st.markdown("---")
st.caption("By Christian Ishimwe – Rwanda's MLOps Champion")