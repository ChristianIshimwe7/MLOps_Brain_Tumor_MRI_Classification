# Brain Tumor MRI Classifier – Full MLOps Pipeline

**Medical Image Classification** | Tumor vs No Tumor | Live Retraining | Docker | Locust Load Test

**Video Demo (Camera ON)**: https://youtu.be/brain-tumor-demo-2025  
**Live UI**: http://localhost:8501  
**API**: http://localhost:8000/docs

## Features
- Real brain MRI images (Kaggle)
- Fine-tuned ResNet-18 (pretrained)
- Single image prediction
- Bulk upload + live retraining
- Streamlit UI with uptime & visualizations
- Docker + Locust scaling test

## Quick Start
```bash
source venv/bin/activate
uvicorn src.api:app --reload --port 8000
streamlit run ui/app.py