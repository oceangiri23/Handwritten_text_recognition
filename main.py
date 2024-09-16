from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import io
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import utils
from segmenter import makesegmentedimage



app = FastAPI()

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:3000",  # Replace with your frontend URL
    "http://127.0.0.1:5500",
    "http://localhost:8000",  # Replace with your backend URL if separate
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500/index.html"
    # Add more origins as needed
]

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# model = utils.build_model()

# Load the model
# custom_objects = {
#     'Orthogonal': tf.keras.initializers.Orthogonal,
# }

# Load the model


loaded_model = tf.keras.models.load_model('model_V50.h5')
class GenerateRequest(BaseModel):
    additional_info: Optional[str] = None


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), additional_info: Optional[str] = Form(None)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are allowed.")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    image.save(file_path)  
    final_text=[]
    makesegmentedimage(file_path, 'segmented_images')
    file_list=os.listdir('segmented_images')
    sorted_files = utils.sort_natural(file_list)
    for i in sorted_files:
        processed_image = utils.processed(os.path.join('segmented_images',i))
        if processed_image.shape != (128, 32, 1):
            raise ValueError("Processed image has incorrect shape. Expected shape is (128, 32, 1).")
        processed_image = np.expand_dims(processed_image, axis=0)
        text = utils.recongize_text_from_already_segmented_image(processed_image,loaded_model)
        final_text+=text+[' ']
    # processed_image = utils.processed(file_path)
    # if processed_image.shape != (128, 32, 1):
    #     raise ValueError("Processed image has incorrect shape. Expected shape is (128, 32, 1).")
    # processed_image = np.expand_dims(processed_image, axis=0)



    # text = utils.recongize_text_from_already_segmented_image(processed_image,loaded_model)
    # Make prediction
    # prediction = loaded_model.predict(processed_image)
    # pred_text=utils.decode_batch_prediction(prediction)
    complete_text=''.join(final_text)
    
    return {
        "filename": file.filename,
        "prediction":complete_text,  # Convert prediction to list for JSON serialization
        "additional_info": additional_info,
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to Handwritten Recognizer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


