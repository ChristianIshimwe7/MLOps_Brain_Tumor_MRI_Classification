import os, sqlite3, torch, boto3
from torch.utils.data import DataLoader, TensorDataset
from src.preprocessing import preprocess_image
from src.model import load_model

DB = "uploads/metadata.db"
s3 = boto3.client('s3')
BUCKET = "mlops-demo-data"

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, s3_key TEXT, label INTEGER)""")
    conn.close()

def save_uploaded(file_bytes: bytes, filename: str, label: int):
    key = f"raw/{filename}"
    s3.put_object(Bucket=BUCKET, Key=key, Body=file_bytes)
    conn = sqlite3.connect(DB)
    conn.execute("INSERT INTO images (s3_key, label) VALUES (?,?)", (key, label))
    conn.commit(); conn.close()

def trigger_retraining():
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT s3_key, label FROM images").fetchall()
    conn.close()
    if not rows: return

    imgs, labels = [], []
    for key, lbl in rows:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        tensor = preprocess_image(obj['Body'].read())
        imgs.append(tensor.squeeze(0))
        labels.append(lbl)

    dataset = TensorDataset(torch.stack(imgs), torch.LongTensor(labels))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-5)
    crit = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

    new_path = "models/resnet_finetuned_latest.pth"
    torch.save(model.state_dict(), new_path)
    s3.upload_file(new_path, BUCKET, "models/resnet_finetuned_latest.pth")
    print("Retraining complete")
