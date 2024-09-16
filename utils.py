import os
import tensorflow as tf
# from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
needed=['[UNK]',
 '-',
 'r',
 'w',
 'S',
 'D',
 'Z',
 'Q',
 'J',
 'I',
 '9',
 '0',
 'j',
 'F',
 'o',
 'L',
 'n',
 'f',
 'M',
 'g',
 'e',
 'Y',
 'u',
 'R',
 'q',
 ':',
 'P',
 'V',
 ';',
 'A',
 '?',
 '+',
 'a',
 'b',
 'K',
 '4',
 'B',
 'h',
 'O',
 'N',
 '&',
 ',',
 't',
 'C',
 '/',
 'd',
 'H',
 'T',
 '#',
 '5',
 '3',
 '6',
 '"',
 's',
 'l',
 'E',
 'x',
 'c',
 '7',
 'p',
 'z',
 'W',
 'X',
 ')',
 "'",
 '1',
 '2',
 'y',
 '*',
 '8',
 '(',
 '!',
 'U',
 'G',
 'k',
 '.',
 'i',
 'v',
 'm']


mapping=['[UNK]',
 '.',
 '9',
 'I',
 '8',
 'j',
 'y',
 'q',
 'Y',
 'T',
 'N',
 'z',
 '&',
 't',
 'h',
 '!',
 '0',
 '2',
 'X',
 'V',
 '4',
 '+',
 ')',
 'W',
 'E',
 'b',
 'J',
 '?',
 'Q',
 'B',
 'R',
 'v',
 'm',
 'P',
 ',',
 '/',
 'M',
 'c',
 'D',
 '#',
 'G',
 'g',
 'e',
 'l',
 'K',
 'o',
 'p',
 "'",
 'a',
 '7',
 '"',
 'u',
 'A',
 '(',
 'w',
 'r',
 ';',
 'x',
 'C',
 'S',
 'i',
 'H',
 '-',
 'k',
 '*',
 '1',
 'U',
 'L',
 'd',
 '5',
 '3',
 '6',
 's',
 'F',
 ':',
 'O',
 'f',
 'Z',
 'n']
char_to_index_dict = {char: idx for idx, char in enumerate(mapping)}

index_to_char_dict = {idx: char for idx, char in enumerate(mapping)}

def char_to_index(char):
    return char_to_index_dict.get(char, char_to_index_dict['[UNK]'])  # Return [UNK] index if char not found

def index_to_char(index):
    return index_to_char_dict.get(index, '[UNK]')
np.random.seed(42)

tf.random.set_seed(42)
batch_size = 1
padding_token = 99
image_width = 128
image_height = 32
characters = {'5', 'o', 'e', 'T', 'f', 'N', 'h', '!', 'S', '1', 'm', 'b', 'c', '.', 'q', 'U', '-', 'K', 'k', 'I', 'M', '2', 'W', 'Q', 'a', 'H', 'P', 'd', '?', '&', '*', 'V', 'R', 'w', 'r', ')', 'l', 'J', '/', 'D', 'i', 't', 'v', 'Y', 'A', 'E', '0', 'B', '"', ';', 'p', '(', '+', 'L', ',', "'", 'j', 'n', 'C', '3', '9', 'g', '4', 'F', '8', 'G', 'x', 'Z', 'y', '#', 'X', '7', 's', ':', 'z', 'O', '6', 'u'}

max_len=21

def sort_natural(file_list):
    """
    Sort a list of filenames in natural order, i.e., numerically when filenames contain numbers.
    """
    def natural_key(text):
        # Split text into a list of text and numbers
        return [int(text) if text.isdigit() else text for text in re.split('([0-9]+)', text)]

    return sorted(file_list, key=natural_key)

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)  # size parameter takes height first and then width
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do some amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    # because tf.resize uses (h, w) way
    return image

def processed(image_path):
    image = preprocessing_image(image_path)
    print(image.shape)
    return image


def preprocessing_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)  # decode the png_encoded images into tensor, channel 1 for gray scale
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0  # data type conversion in tensor
    return image







def decode_batch_prediction(pred,):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    # Use greedy search. For complex tasks, consider using beam search.
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]
    print("the results are", results)
    
    # Initialize the list to store decoded texts
    output_text = []

    for res in results:
        # Convert the Tensor to NumPy array
        res_np = res.numpy()
        
        # Ensure res_np is not empty and has the expected shape
        if res_np.size == 0:
            output_text.append("")
            continue
        
        # Flatten the array if necessary
        if len(res_np.shape) > 1:
            res_np = np.squeeze(res_np)
        
        # Convert to list of indices
        res_int = res_np.astype(int)
        
        # Ensure res_int is a 1D array
        if res_int.ndim != 1:
            res_int = res_int.flatten()
        
        # Convert indices to characters
        res_chars = [index_to_char(int(idx)) for idx in res_int if idx != -1]
        
        # Join the characters to form the final string
        res_str = ''.join(res_chars)
        output_text.append(res_str)

    return output_text


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image = cv2.bitwise_not(binary_image)  # Ensure text is black on white
    return binary_image

def segment_lines(binary_image):
    horizontal_projection = np.sum(binary_image, axis=1)
    threshold = np.max(horizontal_projection) / 2
    lines = []
    start = 0
    in_line = False
    for i, value in enumerate(horizontal_projection):
        if value > threshold and not in_line:
            start = i
            in_line = True
        elif value <= threshold and (in_line and i - start >= 60):
            end = i
            in_line = False
            lines.append((start, end))
    return lines
def invert_colors(image):
    return cv2.bitwise_not(image)
def segment_words(line_image):
    vertical_projection = np.sum(line_image, axis=0)
    threshold = np.max(vertical_projection) / 1.5
    words = []
    start = 0
    in_word = False
    for i, value in enumerate(vertical_projection):
        if value > threshold and not in_word:
            start = i
            in_word = True
        elif value <= threshold and in_word:
            end = i
            in_word = False
            words.append((start, end))
    return words
class DummyModel:
    def predict(self, word_image):
        return "word"

# def recognize_text_in_lines(image_path, model):
#     binary_image = preprocess_image(image_path)
#     segmented_lines = [binary_image[start:end, :] for start, end in segment_lines(binary_image)]
#     full_text = ""
#     word_count=0
#     for line_image in segmented_lines:
#         line_image= invert_colors(line_image)
#         words = segment_words(line_image)
#         line_text = ""
#         for start, end in words:
#             word_image = line_image[:, start:end]
#             word_image_path = f'uploads/word_{word_count + 1}.png'
#             cv2.imwrite(word_image_path, word_image)
#             word_image=processed(word_image_path)
#             word_count += 1
#             word_text = model.predict(np.expand_dims(word_image, axis=0))  # Predict
#             print(word_text)  
#             word_text=decode_batch_prediction(word_text)

#             line_text += word_text[0] + " "
#         full_text += line_text.strip() + "\n"
#     return full_text

def recongize_text_from_already_segmented_image(image,model):
    word_text = model.predict(image)  
    word_text=decode_batch_prediction(word_text)
    return word_text



def display_segmented_lines(segmented_lines):
    for i, line_image in enumerate(segmented_lines):
        
        plt.figure(figsize=(10, 2))
        plt.imshow(line_image, cmap='gray')
        plt.title(f'Segmented Line {i + 1}')
        plt.axis('off')
        plt.show()

# Example usage
image_path = 'uploads/Handwriting.png'

# Display segmented lines


# Recognize text using a dummy model
# dummy_model = DummyModel()
# recognized_text = recognize_text_in_lines(segmented_lines, dummy_model)
# print(recognized_text)