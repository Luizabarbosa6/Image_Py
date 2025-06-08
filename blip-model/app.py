from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch
from io import BytesIO

app = Flask(__name__)

# Usa o modelo diretamente do Hugging Face (sem diretório local)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/caption', methods=['POST'])
def caption_image():
    data = request.get_json()
    image_url = data.get('image_url')

    try:
        image = Image.open(BytesIO(requests.get(image_url).content)).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return jsonify({'caption': caption})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
