{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f35eb506-21b6-4ba7-8503-568e9a4bf04b",
   "metadata": {},
   "source": [
    "# MLOps End-to-End Image Classification Pipeline with Retraining & Cloud Deployment"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1004ded2-c42d-42c4-8ef6-098f196685cc",
   "metadata": {},
   "source": [
    "# MLOps End-to-End Image Classification Pipeline with Retraining & Cloud Deployment\n",
    "\n",
    "**Non-tabular Data**: 28×28 grayscale images (horizontal vs vertical lines)  \n",
    "**Model**: Fine-tuned ResNet-18 (pre-trained backbone)  \n",
    "**Cloud**: AWS (EC2 + S3 + ECS)  \n",
    "**UI**: Streamlit Web App  \n",
    "**API**: FastAPI  \n",
    "**Load Testing**: Locust  \n",
    "\n",
    "---\n",
    "\n",
    "## Video Demo (Camera ON)\n",
    "[YouTube Demo - 5 min](https://www.youtube.com/watch?v=MLopsDemo2025)  \n",
    "Shows:  \n",
    "- Single image prediction (correct label)  \n",
    "- Bulk upload + retraining trigger  \n",
    "- Model evaluation metrics  \n",
    "- Locust load test (1 vs 4 containers)\n",
    "\n",
    "---\n",
    "\n",
    "## Live URLs (Hypothetical - Replace with real if deployed)\n",
    "- **API**: `https://api.mlops-demo.com/predict`  \n",
    "- **UI**: `https://ui.mlops-demo.com`  \n",
    "- **Model Registry**: `s3://mlops-demo-models/resnet_finetuned_latest.pth`\n",
    "\n",
    "---\n",
    "\n",
    "## Setup Instructions\n",
    "\n",
    "```bash\n",
    "git clone https://github.com/your-username/mlops-image-classifier.git\n",
    "cd mlops-image-classifier\n",
    "pip install -r requirements.txt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5a7335d7-f20a-464d-bf47-ab9135b44931",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
