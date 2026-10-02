from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

import os
import uuid
import base64

load_dotenv()

app = Flask(__name__)

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the .env file.")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

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

        # Generate image using OpenAI
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            quality="low"
        )

        # Get generated image data
        image_data = response.data[0].b64_json

        if not image_data:
            return jsonify({
                'error': 'No image was returned.'
            }), 500

        # Create unique filename
        filename = f"{uuid.uuid4().hex}.png"

        filepath = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        # Decode and save image
        with open(filepath, "wb") as image_file:
            image_file.write(base64.b64decode(image_data))

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