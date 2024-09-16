import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

def empty_folder(folder_path):
    """
    Delete all files in the specified folder.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def makesegmentedimage(image_path, output_folder):
    if os.path.exists(output_folder):
        empty_folder(output_folder)
    else:
        os.makedirs(output_folder)
    

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


    # Apply thresholding
    threshold_value = 100
    _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Tesseract configuration
    custom_oem_psm_config = r'--oem 3 --psm 6'

    # Extract data with Tesseract
    data = pytesseract.image_to_data(image, config=custom_oem_psm_config, output_type=pytesseract.Output.DICT)

    word_info = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > -1 and data['text'][i].strip():  
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            word_info.append({
                'index': i,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'text': data['text'][i]
            })

    # Calculate dynamic row height threshold
    ys = [w['y'] for w in word_info]
    median_y = np.median(ys)
    std_y = np.std(ys)
    row_threshold = std_y  # Adjust based on the spread of y values

    word_info.sort(key=lambda w: (w['y']))

    rows = []
    current_row = []

    # Group words into rows based on dynamic threshold
    for word in word_info:
        if current_row and abs(word['y'] - current_row[-1]['y']) > row_threshold:
            rows.append(current_row)
            current_row = []
        current_row.append(word)
    if current_row:
        rows.append(current_row)

    # Sort each row by horizontal position
    for row in rows:
        row.sort(key=lambda w: w['x'])

    cropped_words_images = []
    for row in rows:
        for info in row:
            x, y, w, h = info['x'], info['y'], info['w'], info['h']
            cropped_word = image[y:y+h, x:x+w]
            cropped_words_images.append(cropped_word)

            file_name = f"word_{info['index']+1}.png"
            file_path = os.path.join(output_folder, file_name)
            cv2.imwrite(file_path, cropped_word)  # Save the image
            print(f"Saved {file_path}")

    # Visualize the cropped words
    # if cropped_words_images:
    #     num_images = len(cropped_words_images)
    #     fig, axes = plt.subplots(1, num_images, figsize=(50, 20))  # Adjust figsize as needed

    #     for i, word_image in enumerate(cropped_words_images):
    #         axes[i].imshow(word_image, cmap='gray')
    #         axes[i].axis('off')  # Hide axes for a cleaner display

    #     plt.show()
