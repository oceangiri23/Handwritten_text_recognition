from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

model =YOLO("./model/model.pt")


def sort_boxes_by_rows_dynamic_threshold(boxes):
    """
    Sort bounding boxes first by y_min, grouping them into rows based on a dynamic threshold.
    The dynamic threshold is based on the average height of nearby boxes.
    Then sort each row by x_min.
    
    Parameters:
    boxes (numpy.ndarray): Array of bounding boxes with shape (n, 4) where each box is [x_min, y_min, x_max, y_max].
    
    Returns:
    sorted_boxes (list): List of bounding boxes sorted first by rows and then by x_min within each row.
    """
    # Sort by y_min first
    boxes = sorted(boxes, key=lambda x: x[1])
    
    rows = []
    current_row = [boxes[0]]
    
    for box in boxes[1:]:
        _, y_min, _, y_max = box
        _, prev_y_min, _, prev_y_max = current_row[-1]
        
        # Calculate the height of the previous box and current box
        prev_box_height = prev_y_max - prev_y_min
        current_box_height = y_max - y_min
        
        # Use the average height of the current row as the dynamic threshold
        avg_height = (prev_box_height + current_box_height) / 2
        
        # Check if the current box belongs to the same row (within the range of dynamic threshold)
        if abs(y_min - prev_y_min) <= avg_height:
            current_row.append(box)
        else:
            # Sort the current row by x_min and add it to the rows
            rows.append(sorted(current_row, key=lambda x: x[0]))  # Sort by x_min
            current_row = [box]

    # Don't forget to add the last row
    rows.append(sorted(current_row, key=lambda x: x[0]))

    # Flatten the list of rows to get the final sorted list of boxes
    sorted_boxes = [box for row in rows for box in row]

    return sorted_boxes

def save_detected_image(image_path, output_folder):
    """
    Detect text in the image and save each word as a separate image.
    
    Parameters:
    image_path (str): Path to the input image.
    output_folder (str): Path to the output folder to save the detected words.
    """
    # Load the image
    img = cv2.imread(image_path)
    
    # Perform text detection
    results = model(image_path)
    
    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.numpy()
    
    # Sort the bounding boxes by rows
    sorted_boxes = sort_boxes_by_rows_dynamic_threshold(boxes)
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    list_img=os.listdir(output_folder)

    if(len(list_img)!=0):
        for img_ in list_img:
            os.remove(f'{output_folder}/{img_}')
    
    # Save each word as a separate image
    for i, box in enumerate(sorted_boxes):
        # Extract coordinates (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = map(int, box)
        
        # Crop the image using the bounding box coordinates
        cropped_img = img[y_min:y_max, x_min:x_max]
        
        # Save the cropped image to a folder
        cropped_img_path = os.path.join(output_folder, f"word_{i}.png")
        cv2.imwrite(cropped_img_path, cropped_img)
        
        print(f"Word {i} saved to {cropped_img_path}")

