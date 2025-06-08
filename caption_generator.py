from fastapi import FastAPI
from pydantic import BaseModel
from caption_generator import gerar_caption

app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str

@app.post("/caption")
def caption_image(req: ImageRequest):
    return gerar_caption(req.image_url)

# 🔁 Rodar localmente ou no Render
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
