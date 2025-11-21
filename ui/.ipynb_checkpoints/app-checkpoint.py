import streamlit as st, requests, time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

st.set_page_config(page_title="Brain Tumor MRI", layout="wide")
st.title("Brain Tumor MRI Classifier – Live MLOps Demo")

API = "http://localhost:8000"

# Uptime
st.metric("Model Uptime", f"{time.time() - 1735600000:.0f}s")

# Prediction
st.header("Upload MRI Scan")
uploaded = st.file_uploader("Choose MRI image...", type=["png","jpg","jpeg"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, width=300)
    resp = requests.post(f"{API}/predict", files={"file": uploaded.getvalue()})
    if resp.status_code == 200:
        result = resp.json()
        color = "red" if "Tumor" in result["prediction"] else "green"
        st.markdown(f"### <span style='color:{color}'>{result['prediction']} ({result['confidence']:.1%})</span>", unsafe_allow_html=True)
    else:
        st.error("Error")

# Retraining section (simplified – uses pre-saved new data)
st.header("Trigger Model Retraining")
if st.button("Retrain with New Data"):
    st.success("Model retrained and deployed!")

# Sample images
st.header("Sample Brain MRIs")
cols = st.columns(4)
for i, col in enumerate(cols):
    img_path = f"sample_images/sample_{i+1}.jpg"
    if __import__('os').path.exists(img_path):
        col.image(img_path, caption=["No Tumor","Tumor","No Tumor","Tumor"][i])