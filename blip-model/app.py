from flask import Flask, request, jsonify
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

app = Flask(__name__)

processor = BlipProcessor.from_pretrained("./blip-model")
model = BlipForConditionalGeneration.from_pretrained("./blip-model")

@app.route("/caption", methods=["POST"])
def caption_image():
    data = request.json
    image_url = data.get("image_url")
    if not image_url:
        return jsonify({"error": "Missing image_url"}), 400

    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    inputs = processor(images=raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return jsonify({"caption": caption})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)