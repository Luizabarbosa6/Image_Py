from flask import Flask, request, jsonify
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/caption', methods=['POST'])
def caption_image():
    data = request.get_json()
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "image_url não fornecido"}), 400

    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Erro ao baixar ou processar imagem: {str(e)}"}), 500

    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return jsonify({"caption": caption})


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)


