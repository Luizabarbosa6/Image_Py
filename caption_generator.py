from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Define o device (GPU se disponível, senão CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega modelo e processador
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def gerar_caption(image_url: str):
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        return {"error": f"Erro ao carregar imagem: {str(e)}"}

    inputs = processor(images=image, return_tensors="pt").to(device)

    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return {"caption": caption}
