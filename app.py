from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO

app = FastAPI()

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

class ImageRequest(BaseModel):
    url: str

@app.post("/caption")
async def generate_caption(data: ImageRequest):
    try:
        response = requests.get(data.url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return {"caption": caption}
    except Exception as e:
        return {"error": str(e)}
