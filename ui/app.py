import streamlit as st, requests, time, os, boto3, sqlite3
from src.retrain import init_db, save_uploaded, trigger_retraining

API = "http://localhost:8000"
st.set_page_config(page_title="MLOps Classifier", layout="wide")
st.title("MLOps Image Classifier – Live Pipeline")

# Uptime
start_time = time.time()
st.metric("Model Uptime", f"{(time.time()-start_time)/60:.1f} min")

# Prediction
st.header("Predict Single Image")
uploaded = st.file_uploader("Upload image", type=["png","jpg"])
if uploaded:
    bytes_ = uploaded.read()
    st.image(bytes_, width=150)
    resp = requests.post(f"{API}/predict", files={"file": ("img", bytes_)})
    if resp.status_code == 200:
        pred = resp.json()
        st.success(f"**{pred['prediction'].title()}** – {pred['confidence']:.1%} confidence")
    else:
        st.error("Prediction failed")

# Retraining
st.header("Bulk Upload & Retrain")
files = st.file_uploader("Upload labeled images", accept_multiple_files=True)
label = st.selectbox("Label", [0,1], format_func=lambda x: ["Horizontal","Vertical"][x])
col1, col2 = st.columns(2)
with col1:
    if st.button("Save Data") and files:
        init_db()
        for f in files:
            save_uploaded(f.read(), f.name, label)
        st.success(f"{len(files)} images saved")
with col2:
    if st.button("Trigger Retraining"):
        with st.spinner("Retraining..."):
            trigger_retraining()
        st.success("New model deployed!")

# Visualizations
st.header("Dataset Insights")
if os.path.exists("data/train"):
    import numpy as np, matplotlib.pyplot as plt
    samples = [np.load(f"data/train/img_{i}.npy").squeeze() for i in range(9)]
    fig, axs = plt.subplots(3,3,figsize=(6,6))
    for ax, img in zip(axs.flat, samples):
        ax.imshow(img, cmap='gray'); ax.axis('off')
    st.pyplot(fig)
