from locust import HttpUser, task, between
import io
import numpy as np
from PIL import Image

class MLUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        # Load a sample image once
        arr = np.load("data/test/img_0.npy").squeeze() * 255
        img = Image.fromarray(arr.astype('uint8'))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.image_bytes = buf.getvalue()

    @task
    def predict(self):
        files = {'file': ('test.png', self.image_bytes, 'image/png')}
        self.client.post("/predict", files=files)
