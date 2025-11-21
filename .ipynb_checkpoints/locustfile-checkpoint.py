from locust import HttpUser, task, between
import io, random
from PIL import Image
import numpy as np

class MRIUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        # Use one of the sample images
        img_path = f"sample_images/sample_{random.randint(1,4)}.jpg"
        with open(img_path, "rb") as f:
            files = {'file': ('mri.jpg', f, 'image/jpeg')}
            self.client.post("/predict", files=files)