import streamlit as st
import requests
from PIL import Image

st.title("Brain Tumor MRI Classifier")
uploaded = st.file_uploader("Upload MRI", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, width=350)
    with st.spinner("Analyzing..."):
        try:
            r = requests.post("http://127.0.0.1:8000/predict", files={"file": uploaded.getvalue()})
            result = r.json()
            if "Tumor" in result["prediction"]:
                st.error(f"**{result['prediction']}**")
                st.warning(f"Confidence: {result['confidence']:.1%}")
            else:
                st.success(f"**{result['prediction']}**")
                st.balloons()
                st.info(f"Confidence: {result['confidence']:.1%}")
        except:
            st.error("Start backend: uvicorn src.api:app --port 8000")