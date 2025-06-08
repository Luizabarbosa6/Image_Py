import sys
import base64
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Carrega modelo
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Lê a imagem em base64 recebida pelo stdin
image_base64 = sys.stdin.read()
image_data = base64.b64decode(image_base64)
image = Image.open(io.BytesIO(image_data)).convert("RGB")

# Processa a imagem
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)

# Gera legenda
caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)
