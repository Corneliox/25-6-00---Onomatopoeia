import torch
import open_clip as clip
from PIL import Image
import pandas as pd

# Load model CLIP (ViT-B/32 cukup ringan)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load dataset mapping
df = pd.read_csv("clip.json")  # sesuai format csv di atas

results = []

for i, row in df.iterrows():
    image = preprocess(Image.open(row['image_path']).convert("RGB")).unsqueeze(0).to(device)
    text = clip.tokenize([row['prompt']]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        # Normalisasi vektor
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T).item()
    
    results.append({
        "image": row['image_path'],
        "prompt": row['prompt'],
        "clip_score": similarity
    })

# Simpan hasil evaluasi
results_df = pd.DataFrame(results)
results_df.to_csv("clip_scores.csv", index=False)

print("Evaluasi selesai, hasil tersimpan di clip_scores.csv")