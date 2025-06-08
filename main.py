from fastapi import FastAPI, UploadFile, File
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

app = FastAPI()

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.post("/caption")
async def generate_caption(file: UploadFile = File(...)):
    image = Image.open(await file.read()).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return {"caption": caption}
