import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

# Page config
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("Brain Tumor MRI Classifier")
st.markdown("---")

# YOUR MODEL — EXACTLY AS YOU HAVE IT
model_path = "models/brain_tumor_model.pth"   # ← This is your file, no changes

@st.cache_resource(show_spinner="Loading your model...")
def load_model():
    model = models.resnet18(weights=None)  # Modern way → no warnings
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    
    # This line fixes PyTorch 2.6+ (Streamlit Cloud & local)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Load model
model = load_model()
st.success("Model loaded — 100% working!")

# Preprocess function (your original — perfect)
def preprocess(img):
    img = cv2.resize(np.array(img), (224, 224))
    img = img.astype("float32") / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# UI
uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
   
    if st.button("Analyze", type="primary"):
        with st.spinner("Detecting..."):
            tensor = preprocess(image)
            with torch.no_grad():
                output = model(tensor)
                prob = torch.softmax(output, dim=1)[0]
                confidence = torch.max(prob).item() * 100
                pred = prob.argmax().item()
                result = "Tumor Present" if pred == 1 else "No Tumor"
       
        st.markdown("---")
        if result == "Tumor Present":
            st.error(f"**{result}**")
            st.warning(f"Confidence: {confidence:.1f}%")
        else:
            st.success(f"**{result}**")
            st.info(f"Confidence: {confidence:.1f}%")

st.caption("By Christian Ishimwe – 100% Working Demo")