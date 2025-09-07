from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP-2 VQA model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to(device)

# Load pertanyaan dari file JSON
with open("vqa_questions.json", "r", encoding="utf-8") as f:
    vqa_data = json.load(f)

results = []

for item in vqa_data:
    image = Image.open(item["image"]).convert("RGB")
    
    for q in item["questions"]:
        inputs = processor(image, q, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)
        
        results.append({
            "image": item["image"],
            "question": q,
            "answer": answer
        })

import pandas as pd
df = pd.DataFrame(results)
df.to_csv("vqa_results.csv", index=False)

print("Evaluasi VQA selesai, hasil tersimpan di vqa_results.csv")