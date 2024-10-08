from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import io
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
import os
import utils
from detect import save_detected_image
import requests
from spellchecker import SpellChecker
import cv2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
spell = SpellChecker()


""" 
CORS settings to allow requests from specific origins such as localhost for local development. 
This middleware ensures that the frontend can communicate with the backend API.
"""
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

""" 
Load a pre-trained TensorFlow model for recognizing text from handwritten images.
The model is loaded once when the server starts to avoid reloading for each request.
"""
loaded_model = tf.keras.models.load_model('./model/model_V80.h5')


model = SentenceTransformer('bert-base-nli-mean-tokens')



class ImageURLRequest(BaseModel):
    """
    A request model that accepts an image URL and optional additional information.
    """
    image_url: str
    additional_info: Optional[str] = None

def resize_and_pad_image(image,average_color, target_size=1024):
    """
    Resizes and pads the image to a square of target size (1024x1024 by default).
    If the image is smaller, it pads it with the average color of the image.
    """
    # Get the current size of the image
    width, height = image.size

    # If the image is smaller than the target size, pad it with the padding color
    if width < target_size or height < target_size:
        # Calculate the padding needed to reach the target size
        pad_width = max(target_size - width, 0)
        pad_height = max(target_size - height, 0)

        # Create a new image with the target size and the padding color (white in this case)
        padded_image = ImageOps.expand(image, border=(0, 0, pad_width, pad_height), fill=average_color)
        
        return padded_image
    else:
        # If the image is already larger or equal to the target size, return it as-is
        return image


@app.post("/upload-image")
async def upload_image_url(request: ImageURLRequest):
    """
    Endpoint to upload an image via URL, process it, and extract text using a pre-trained TensorFlow model.
    - Downloads the image from the provided URL.
    - Resizes and pads the image.
    - Passes the processed image through a text recognition pipeline.
    - Returns the recognized text.

    Args:
    request (ImageURLRequest): Contains the image URL and optional additional information.
    
    Returns:
    dict: A dictionary with the filename and predicted text.
    """
    import time
    start_time = time.time()

    image_url = request.image_url

    if not image_url.lower().startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400, detail="Invalid URL. The URL must start with http:// or https://.")
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Ensure the request was successful
        image = Image.open(io.BytesIO(response.content))
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400, detail=f"Error downloading the image: {str(e)}")
    
    if image.mode in ("P", "RGBA"):
        image = image.convert("RGB")
    image.save('test.png')

# Check dimensions and handle both grayscale and RGB cases
    image_array = np.array(image)
    average_color = np.mean(image_array, axis=(0, 1))

    if image_array.ndim == 3 and average_color.shape[0] == 3:
        average_color = tuple(map(int, average_color))  # RGB case
    else:
        average_color = int(average_color)
    image= resize_and_pad_image(image,average_color=average_color)
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, 'uploaded_image.jpg')
    image.save(file_path)
    final_text = []
    save_detected_image(file_path, 'segmented_images',average_color,upload_dir)
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
    for i in range(len(final_text)):
        if final_text[i] not in [' ', '', '\n', '"']:
            temp = spell.correction(final_text[i])
            if temp:
                final_text[i] = temp

    complete_text = ''.join(final_text)
    end_time = time.time()

    return {
        "filename": 'temo.jpg',
        "prediction": complete_text,  
    }


class GradeReq(BaseModel):
    base_text: str
    obtained_text: str



def predict_similarity(base_text,obtained_text):
    """
    Computes the similarity score between the base text and obtained text using cosine similarity on sentence embeddings.

    Args:
    - base_text (str): The original reference text.
    - obtained_text (str): The text to be compared against the base text.

    Returns:
    - float: The cosine similarity score between the base text and obtained text. 
             A score of 1.0 means perfect similarity, while 0 means no similarity.

    Steps:
    1. The two input texts (base_text, obtained_text) are converted into embeddings using a pre-trained model (`model.encode`).
    2. Cosine similarity is calculated between the embeddings of the base text and the obtained text.
    """
    sentences=[
        base_text,obtained_text,''
    ]    
    sentence_embeddings = model.encode(sentences)
    similarity_scores = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])

    return similarity_scores[0][0]



@app.post("/grade")
def grade(request: GradeReq):
    """
    Endpoint to grade the accuracy of recognized text by comparing it with the base text.
    - Compares both texts word by word.
    - Calculates and returns the accuracy as a ratio of correct words.

    Args:
    request (GradeReq): Contains the base text and the recognized text.

    Returns:
    dict: A dictionary containing the accuracy score.
    """
    base_text = request.base_text
    obtained_text = request.obtained_text
    print(base_text,obtained_text)

    result =predict_similarity(base_text,obtained_text)
    result*=100
    result=int(result)
    return {
        "result":result
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to Handwritten Recognizer"}


if __name__ == "__main__":
    """
    Runs the FastAPI application with Uvicorn on localhost and port 8000.
    """
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
