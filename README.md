# Handwritten Text Recognition

This project focuses on recognizing handwritten text using two different models: one for word recognition and another for word detection. The models are trained on the IAM dataset.

## Project Structure

- **Backend**: A FastAPI backend (`main.py`) that handles model inference.
- **Frontend**: A Next.js application for the user interface.
- **Models**: 
  1. **Word Recognition Model**: A CNN+LSTM+CTC architecture for recognizing words from images.
  2. **Word Detection Model**: A YOLOv8 model for detecting word boundaries in images.

## Features

- Detects word boundaries in handwritten text images using a YOLOv8 model.
- Recognizes words within the detected boundaries using a CNN+LSTM+CTC model.

## Dataset

The models are trained on the **IAM Handwriting Dataset**, which contains handwritten text lines and corresponding ground truth annotations.

## Installation

### Prerequisites

- Python 
- Node.js (for frontend)
- pip (Python package manager)

### Backend Setup

1. Clone the repository:

    ```bash
    git clone git remote add origin https://github.com/nirajan1111/Handwritten_Text_Recognition_OCR.git

    cd Handwritten_Text_Recognition_OCR
    ```

2. Set up a virtual environment and install the dependencies:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. Run the FastAPI backend:

    ```bash
    uvicorn main:app --reload
    ```

   The backend will be available at `http://127.0.0.1:8000`.

### Frontend Setup

1. Navigate to the `frontend` folder:

    ```bash
    cd frontend
    ```

2. Install frontend dependencies:

    ```bash
    npm install
    ```

3. Run the Next.js development server:

    ```bash
    npm run dev
    ```

   The frontend will be available at `http://localhost:3000`.

## How to Use

1. Upload an image containing handwritten text via the frontend.
2. The backend will detect word boundaries using the YOLOv8 model.
3. For each detected word, the recognition model (CNN+LSTM+CTC) will predict the text.

## Models

- **Word Recognition (CNN+LSTM+CTC)**:
  - The model is trained on the IAM dataset and is responsible for recognizing words from handwritten images.
  
- **Word Detection (YOLOv8)**:
  - This model detects word boundaries and outputs bounding boxes for each word in the input image.

## Acknowledgments
- **IAM Dataset**: The models are trained using the IAM Handwriting Dataset.
- **YOLOv8**: Used for word boundary detection.
- **CNN+LSTM+CTC**: Architecture for word recognition.