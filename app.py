from fastapi import FastAPI
from pydantic import BaseModel
from caption_generator import gerar_caption

app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str

@app.post("/caption")
def caption_image(req: ImageRequest):
    return gerar_caption(req.image_url)
