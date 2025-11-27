# Brain Tumor MRI Classifier – Full MLOps Pipeline

**Medical Image Classification** | Tumor vs No Tumor | Live Retraining | Docker | Locust Load Test

**FULL APP LINK**: https://christianishimwe7-mlops-brain-tumor-mri-classifica-uiapp-upmfgf.streamlit.app/
**Video Demo**: https://youtu.be/5qRDArGu-JQ
**SLIDES PRESENTATION**: https://docs.google.com/presentation/d/1F8HkD0QtbbKIi-M_bVtgL9tAN545Awa9/edit?usp=sharing&ouid=111599858326362392445&rtpof=true&sd=true
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
