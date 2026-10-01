from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

import os
import uuid

load_dotenv()

app = Flask(__name__)

# Get Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set in the .env file.")

# Hugging Face client
client = InferenceClient(
    api_key=HF_TOKEN
)

# Folder for generated images
OUTPUT_FOLDER = "static/generated"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate_image', methods=['POST'])
def generate_image():

    try:

        # Get prompt from browser
        prompt = request.json.get('prompt')

        if not prompt:
            return jsonify({
                'error': 'Please enter a description.'
            }), 400

        print("Generating image...")
        print("Prompt:", prompt)

        # Generate image using Hugging Face
        image = client.text_to_image(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell"
        )

        # Create unique filename
        filename = f"{uuid.uuid4().hex}.png"

        filepath = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        # Save image
        image.save(filepath)

        print("Image generated successfully!")
        print("Saved to:", filepath)

        # URL that browser can access
        image_url = f"/static/generated/{filename}"

        return jsonify({
            'image_url': image_url
        })

    except Exception as e:

        print("Error generating image:")
        print(e)

        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )