# MLOps End-to-End Image Classification Pipeline with Retraining & Cloud Deployment

**Non-tabular Data**: 28×28 grayscale images (horizontal vs vertical lines)  
**Model**: Fine-tuned ResNet-18 (pre-trained backbone)  
**Cloud**: AWS (EC2 + S3 + ECS)  
**UI**: Streamlit Web App  
**API**: FastAPI  
**Load Testing**: Locust  

---

## Video Demo (Camera ON)
[YouTube Demo - 5 min](https://www.youtube.com/watch?v=MLopsDemo2025)  
Shows:  
- Single image prediction (correct label)  
- Bulk upload + retraining trigger  
- Model evaluation metrics  
- Locust load test (1 vs 4 containers)

---

## Live URLs (Hypothetical - Replace with real if deployed)
- **API**: `https://api.mlops-demo.com/predict`  
- **UI**: `https://ui.mlops-demo.com`  
- **Model Registry**: `s3://mlops-demo-models/resnet_finetuned_latest.pth`

---

## Setup Instructions

```bash
git clone https://github.com/ChristianIshimwe7/mlops-image-classifier.git
cd mlops-image-classifier
pip install -r requirements.txt
