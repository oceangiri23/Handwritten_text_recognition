from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import io
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import utils
from detect import save_detected_image
import requests
from spellchecker import SpellChecker

app = FastAPI()
spell = SpellChecker()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500/index.html"
]

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


loaded_model = tf.keras.models.load_model('./model/model_V80.h5')


class GenerateRequest(BaseModel):
    additional_info: Optional[str] = None


class ImageURLRequest(BaseModel):
    image_url: str
    additional_info: Optional[str] = None


@app.post("/upload-image")
async def upload_image_url(request: ImageURLRequest):
    print(request)
    import time
    start_time = time.time()

    image_url = request.image_url
    print(image_url)
    # additional_info = request.additional_info
    # if file.content_type not in ["image/jpeg", "image/png"]:
    #     raise HTTPException(
    #         status_code=400, detail="Invalid file type. Only JPEG and PNG are allowed.")
    if not image_url.lower().startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400, detail="Invalid URL. The URL must start with http:// or https://.")

    try:
        # Download the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Ensure the request was successful
        image = Image.open(io.BytesIO(response.content))
        filename = os.path.basename(image_url)
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400, detail=f"Error downloading the image: {str(e)}")

    if image.mode in ("P", "RGBA"):
        image = image.convert("RGB")
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, 'uploaded_image.jpg')
    image.save(file_path)
    final_text = []
    save_detected_image(file_path, 'segmented_images')
    file_list = os.listdir('segmented_images')
    sorted_files = utils.sort_natural(file_list)
    for i in sorted_files:
        processed_image = utils.processed(os.path.join('segmented_images', i))
        if processed_image.shape != (128, 32, 1):
            raise ValueError(
                "Processed image has incorrect shape. Expected shape is (128, 32, 1).")
        processed_image = np.expand_dims(processed_image, axis=0)
        text = utils.recongize_text_from_already_segmented_image(
            processed_image, loaded_model)
        final_text += text+[' ']
    print(final_text)
    for i in range(len(final_text)):
        if final_text[i] not in [' ', '', '\n', '"']:
            temp = spell.correction(final_text[i])
            if temp:
                final_text[i] = temp
    print(final_text)

    complete_text = ''.join(final_text)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    return {
        "filename": 'temo.jpg',
        "prediction": complete_text,  # Convert prediction to list for JSON serialization
    }


class GradeReq(BaseModel):
    base_text: str
    obtained_text: str


@app.post("/grade")
def grade(request: GradeReq):
    base_text = request.base_text
    obtained_text = request.obtained_text
    base_text = base_text.split()
    obtained_text = obtained_text.split()
    print(base_text)
    print(obtained_text)
    correct = 0
    for i in range(len(base_text)):
        if i < len(obtained_text):
            if base_text[i] == obtained_text[i]:
                correct += 1
    accuracy = correct/len(base_text)
    return {
        "accuracy": accuracy
    }


@app.get("/")
def read_root():
    return {"message": "Welcome to Handwritten Recognizer"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
