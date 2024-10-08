from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

model = YOLO("./model/best100.pt")


def sort_boxes_by_rows_dynamic_threshold(boxes):
    """
    Sort bounding boxes by rows using a dynamic threshold based on average box height.
    The function first sorts by the y_min coordinate to group boxes into rows.
    It then sorts each row by x_min to ensure that words are read left to right within each row.

    Parameters:
    - boxes (numpy.ndarray): Array of bounding boxes with shape (n, 4) where each box is [x_min, y_min, x_max, y_max].

    Returns:
    - sorted_boxes (list): List of bounding boxes sorted first by rows (based on y_min) and then by x_min within each row.
    """
    # Sort by y_min
    boxes = sorted(boxes, key=lambda x: x[1])

    rows = []
    current_row = [boxes[0]]

    for box in boxes[1:]:
        _, y_min, _, y_max = box
        _, prev_y_min, _, prev_y_max = current_row[-1]

        # Calculate heights of current and previous boxes
        prev_box_height = prev_y_max - prev_y_min
        current_box_height = y_max - y_min

        # Dynamic threshold is the average of current and previous box heights
        avg_height = (prev_box_height + current_box_height) / 2

        # Group into the same row if the y_min values are within the dynamic threshold
        if abs(y_min - prev_y_min) <= avg_height:
            current_row.append(box)
        else:
            # Sort current row by x_min and store it
            rows.append(sorted(current_row, key=lambda x: x[0]))
            current_row = [box]

    # Append the last row
    rows.append(sorted(current_row, key=lambda x: x[0]))

    # Flatten list of rows into a single list of sorted boxes
    sorted_boxes = [box for row in rows for box in row]

    return sorted_boxes


def rotate_image(image, average_color):
    """
    Detects the angle of the text in the image and rotates it to be horizontal if necessary.
    The rotation angle is determined based on contours and adjusted using OpenCV transformations.

    Parameters:
    - image (numpy.ndarray): The input image to be rotated.
    - average_color (tuple): Average color of the image to be used as padding if needed.

    Returns:
    - rotated_resized (numpy.ndarray): The rotated and resized image.
    - change_status (bool): A flag indicating if the image was rotated.
    """
    if image is None:
        print("Error: Unable to load the image.")
        return None
    test = cv2.imread("test.png")
    gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to find edges
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Select the contour with the largest area
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    angle = rect[-1]

    # Adjust the angle if it's greater than 90 degrees
    if abs(angle) > 90:
        return image, False

    if abs(angle) > 45:
        if angle > 0:
            angle = 90 - angle
            angle = -angle
        else:
            angle = -90 - angle
   
    print("The angle with the horizontal line is:", angle)

    # Skip rotation if the angle is small
    if abs(angle) < 16:
        return image, False

    # Rotate the image if the angle is reasonable for adjustment
    if abs(angle) < 40:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix and rotate the image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Adjust the rotation matrix to account for the translation
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += new_w / 2 - center[0]
        M[1, 2] += new_h / 2 - center[1]

        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=average_color)
        rotated_resized = cv2.resize(rotated, (1024, 1024), interpolation=cv2.INTER_LINEAR)

        print(rotated_resized.shape)

        # Save and return the rotated image
        cv2.imwrite("rotated_image.jpg", rotated_resized)
        return rotated_resized, True
    else:
        return image, False


def save_detected_image(image_path, output_folder, average_color, upload_dir):
    """
    Detects text from an image, rotates it if necessary, and saves each detected word as a separate image.
    The YOLO model is used to detect the bounding boxes around words in the image.

    Parameters:
    - image_path (str): Path to the input image.
    - output_folder (str): Path to the folder where detected word images will be saved.
    - average_color (tuple): Average color of the image used for padding in rotation.
    - upload_dir (str): Directory where uploaded images are stored.

    Returns:
    None
    """
    # Load the input image
    img = cv2.imread(image_path)
    print(image_path)

    # Rotate the image if necessary
    img, change_status = rotate_image(img, average_color)
    if change_status:
        results = model('rotated_image.jpg')
    else:
        results = model(image_path)

    # Apply YOLO model to detect text
    confidence_scores = results[0].boxes.conf.numpy()
    boxes = results[0].boxes.xyxy.numpy()

    confidence_threshold = 0.55
    high_confidence_mask = confidence_scores >= confidence_threshold
    filtered_boxes = boxes[high_confidence_mask]

    if len(filtered_boxes) == 0:
        print("No text detected.")
        return

    # Sort the bounding boxes by rows and columns
    sorted_boxes = sort_boxes_by_rows_dynamic_threshold(filtered_boxes)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Clear the output folder if there are any pre-existing images
    list_img = os.listdir(output_folder)
    if len(list_img) != 0:
        for img_ in list_img:
            os.remove(f'{output_folder}/{img_}')

    # Save each detected word as a separate image
    for i, box in enumerate(sorted_boxes):
        # Extract coordinates of the bounding box (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = map(int, box)
        print(x_max, x_min, y_max, y_min)
        print("Yup the shape is", img.shape)

        # Crop the image based on the bounding box coordinates
        cropped_img = img[abs(y_min - 5):(y_max + 5), abs(x_min - 5):x_max + 5]

        # Save the cropped image to the output folder
        cropped_img_path = os.path.join(output_folder, f"word_{i}.png")
        cv2.imwrite(cropped_img_path, cropped_img)

        # print(f"Word {i} saved to {cropped_img_path}")
