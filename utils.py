import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re

mapping=['[UNK]',
 "'",
 'Z',
 'G',
 'e',
 'b',
 'E',
 'Q',
 't',
 'd',
 '6',
 'N',
 'X',
 'p',
 '+',
 '-',
 'w',
 'g',
 'j',
 'S',
 'u',
 '8',
 'n',
 '!',
 'B',
 'P',
 'a',
 'o',
 'M',
 'r',
 '5',
 'W',
 'O',
 '3',
 'k',
 'A',
 'h',
 'c',
 'D',
 'i',
 '0',
 'C',
 'q',
 'V',
 'U',
 'T',
 '4',
 '7',
 'v',
 'I',
 'F',
 ')',
 '/',
 'R',
 ',',
 '?',
 'K',
 '#',
 'l',
 '.',
 '1',
 ':',
 'J',
 'x',
 'y',
 ';',
 's',
 'L',
 '*',
 '2',
 '"',
 'Y',
 'z',
 '&',
 'H',
 '9',
 'f',
 '(',
 'm']
char_to_index_dict = {char: idx for idx, char in enumerate(mapping)}

index_to_char_dict = {idx: char for idx, char in enumerate(mapping)}

def char_to_index(char):
    return char_to_index_dict.get(char, char_to_index_dict['[UNK]'])

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
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)  
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

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
    return image

def processed(image_path):
    image = preprocessing_image(image_path)
    print(image.shape)
    return image


def preprocessing_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1) 
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0  
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


