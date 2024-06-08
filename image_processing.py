import joblib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # to prevent RuntimeError: main thread is not in main loop

import matplotlib.pyplot as plt
import math
import multiprocessing
from collections import Counter
from pylab import savefig
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from skimage.morphology import skeletonize
import os
from sklearn.pipeline import make_pipeline
from joblib import dump 

# this function is used to convert the image to grayscale

def binary(image_source, threshold=128, isMatching=False):
    # Open the image
    img = Image.open(image_source).convert('L')  # Convert to grayscale

    # Convert to binary
    binary_img = img.point(lambda x: 0 if x < threshold else 255, '1')  # Thresholding

    # Save the binary image
    if isMatching:
        binary_img.save("static/img/matching/binary_img.jpg")
    else:
        binary_img.save("static/img/img_now.jpg")

def grayscale(image_source, isMatching=False):
    new_img = Image.open(image_source).convert('RGB')

    if not is_grey_scale(image_source):
        img = Image.open(image_source)
        img_arr = np.asarray(img)
        r = img_arr[:, :, 0]
        g = img_arr[:, :, 1]
        b = img_arr[:, :, 2]
        new_arr = r.astype(int) + g.astype(int) + b.astype(int)

        '''
        convert to unsigned int 8-bit because pixel value for grayscale
        is between [0, 255].
        '''
        new_arr = (new_arr/3).astype('uint8')

        new_img = Image.fromarray(new_arr)
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_1.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to check whether the image is grayscale or not
def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size # image dimension, w for width, h for height
    for i in range(w):
        for j in range(h):
            r, g, b = im.getpixel((i, j))
            if r != g != b:
                return False # image is not grayscale
    return True

def dilasi(image_source, kernel_size=5, iterations=1, isMatching=False):
    img = cv2.imread(image_source, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_img = cv2.dilate(img, kernel, iterations=iterations)
    if isMatching:
        cv2.imwrite("static/img/matching/dilated_img.jpg", dilated_img)
    else:
        cv2.imwrite("static/img/img_now.jpg", dilated_img)

def erosi(image_source, kernel_size=5, iterations=1, isMatching=False):
    img = cv2.imread(image_source, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_img = cv2.erode(img, kernel, iterations=iterations)
    if isMatching:
        cv2.imwrite("static/img/matching/eroded_img.jpg", eroded_img)
    else:
        cv2.imwrite("static/img/img_now.jpg", eroded_img)

def opening(image_source, kernel_size=5, iterations=1, isMatching=False):
    img = cv2.imread(image_source, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if isMatching:
        cv2.imwrite("static/img/matching/opened_img.jpg", opened_img)
    else:
        cv2.imwrite("static/img/img_now.jpg", opened_img)

def closing(image_source, kernel_size=5, iterations=1, isMatching=False):
    img = cv2.imread(image_source, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    if isMatching:
        cv2.imwrite("static/img/matching/closed_img.jpg", closed_img)
    else:
        cv2.imwrite("static/img/img_now.jpg", closed_img)
    print("jello")

def countObject(image_source, threshold=128, isMatching=False):
    # Baca gambar
    image = cv2.imread(image_source)
    
    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Menggunakan operasi morfologi untuk menghilangkan noise
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    # Buat mask dengan menggunakan ambang tertentu
    binary_arr = np.where(opening >= threshold, 255, 0).astype('uint8')

    # Deteksi kontur
    contours, _ = cv2.findContours(binary_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Menggambar kotak pembatas untuk setiap kontur
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    
    # Tambahkan teks jumlah serpihan kaca di atas gambar
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"Jumlah: {len(contours)}", (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Menyimpan gambar dengan kotak pembatas
    if isMatching:
        cv2.imwrite("static/img/matching/result.jpg", image)
    else:
        cv2.imwrite("static/img/img_now.jpg", image)
    
    # Menampilkan gambar hasil deteksi

    
    # Mengembalikan jumlah serpihan kaca yang terdeteksi
    return len(contours)


def create_combined_digit_image(digits, filename):
    width, height = 100 * len(digits), 100
    blank_image = np.ones((height, width, 3), np.uint8) * 255

    # Draw each digit
    for i, digit in enumerate(digits):
        text_size = cv2.getTextSize(str(digit), cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
        x = i * 100 + (100 - text_size[0]) // 2
        y = (height + text_size[1]) // 2
        cv2.putText(blank_image, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)

    # Save the image
    combined_digits_folder = 'static/img/combined'
    if not os.path.exists(combined_digits_folder):
        os.makedirs(combined_digits_folder)
    
    combined_image_path = os.path.join(combined_digits_folder, f"{filename}.png")  # Ensure the file has a valid extension
    cv2.imwrite(combined_image_path, blank_image)

    return combined_image_path

# Existing functions
def create_digit_image(digit, filename):
    width, height = 100, 100
    blank_image = np.ones((height, width, 3), np.uint8) * 255

    text_size = cv2.getTextSize(str(digit), cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2

    cv2.putText(blank_image, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)

    digits_folder = 'static/img/digits'
    if not os.path.exists(digits_folder):
        os.makedirs(digits_folder)
    
    digits_image_path = os.path.join(digits_folder, f"{filename}.png")  # Ensure the file has a valid extension
    cv2.imwrite(digits_image_path, blank_image)

def create_digit_image_1_to_10():
    for digit in range(10):
        filename = f'angka_{digit}'
        create_digit_image(digit, filename)

def calculate_freeman_chain_code(contour):
    freeman_chain_code = ""
    directions = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]

    for i in range(1, len(contour)):
        dx = contour[i][0][0] - contour[i-1][0][0]
        dy = contour[i][0][1] - contour[i-1][0][1]

        direction = None
        for idx, (dir_x, dir_y) in enumerate(directions):
            if dx == dir_x and dy == dir_y:
                direction = idx
                break

        if direction is not None:
            freeman_chain_code += str(direction)
    
    return freeman_chain_code

def get_contours(img_arr):
    contours, hierarchy = cv2.findContours(img_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if contours:
        outer_contours = contours[1]
        contour_image = np.zeros_like(img_arr, dtype=np.uint8)
        cv2.drawContours(contour_image, outer_contours, -1, (255, 255, 255), 3)
        return outer_contours, contour_image, contours
    else:
        print("No outer contours found for that digit image!")

def get_contour_from_digit(digitImgPath, use_skeletonize=False):
    gs_image = cv2.imread(f'{digitImgPath}', cv2.IMREAD_GRAYSCALE)

    _, binary_image = cv2.threshold(gs_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if not use_skeletonize:
        outer_contour, contour_image, contours = get_contours(binary_image)
        freeman_code = calculate_freeman_chain_code(outer_contour)
    else:
        binary_image = cv2.bitwise_not(binary_image)
        thinned_image = skeletonize(binary_image).astype(np.uint8) * 255
        contours, _ = cv2.findContours(thinned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        freeman_code = ""
        if contours:
            contour_image = np.zeros_like(thinned_image, dtype=np.uint8)
            cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 3)
            freeman_code += calculate_freeman_chain_code(contours[0])
        else:
            print(f"No contours found for digit")

    return freeman_code, contour_image, contours

def generate_freeman_chain_code_to_env_1(digit):
    if digit == 0:
        with open('metode1.env', 'w') as f:
            f.write("")

    freeman_code_img_a, contour_image, _ = get_contour_from_digit(f'static/img/digits/angka_{digit}.png', use_skeletonize=False)

    with open('metode1.env', 'a') as f:
        f.write(f"{digit} = {freeman_code_img_a};\n")

    contour_folder = 'static/img/contour'
    if not os.path.exists(contour_folder):
        os.makedirs(contour_folder)
    
    contour_image_path = os.path.join(contour_folder, f'contour_image_{digit}.png')
    cv2.imwrite(contour_image_path, contour_image)

def generate_freeman_chain_code_to_env_1_from_1_to_10():
    for digit in range(10):
        generate_freeman_chain_code_to_env_1(digit)

def generate_freeman_chain_code_to_env_2(digit):
    if digit == 0:
        with open('metode2.env', 'w') as f:
            f.write("")

    freeman_code, contour_image, _ = get_contour_from_digit(f'static/img/digits/angka_{digit}.png', use_skeletonize=True)    

    with open('metode2.env', 'a') as f:
        f.write(f"{digit} = {freeman_code};\n")
    
    contour_folder = 'static/img/contour'
    if not os.path.exists(contour_folder):
        os.makedirs(contour_folder)
    
    contour_image_path = os.path.join(contour_folder, f'contour_image_v2_{digit}.png')
    cv2.imwrite(contour_image_path, contour_image)

def generate_freeman_chain_code_to_env_2_from_1_to_10():
    for digit in range(10):
        generate_freeman_chain_code_to_env_2(digit)

def calculate_distance(chain_code1, chain_code2):
    if len(chain_code1) != len(chain_code2):
        if len(chain_code1) > len(chain_code2):
            chain_code2 = '0' * (len(chain_code1) - len(chain_code2)) + chain_code2
        else:
            chain_code1 = '0' * (len(chain_code2) - len(chain_code1)) + chain_code1

    distance = 0
    for symbol1, symbol2 in zip(chain_code1, chain_code2):
        if symbol1 != symbol2:
            distance += 1

    return distance

def guess_digit_shape(digitImgPath, use_skeletonize):
    chain_codes = {}
    if not use_skeletonize:
        with open('metode1.env', 'r') as f:
            for line in f:
                digit, stored_code = line.strip().split('=')
                chain_codes[int(digit.strip())] = stored_code.strip()
    else:
        with open('metode2.env', 'r') as f:
            for line in f:
                digit, stored_code = line.strip().split('=')
                chain_codes[int(digit.strip())] = stored_code.strip()

    freeman_code_chain, contour_image, contours = get_contour_from_digit(digitImgPath, use_skeletonize=use_skeletonize)

    if contours:
        if not use_skeletonize:
            chain_code_input_img = calculate_freeman_chain_code(contours[1])
        else:
            chain_code_input_img = calculate_freeman_chain_code(contours[0])

        best_match = None
        best_distance = float('inf')

        for digit, stored_code in chain_codes.items():
            distance = calculate_distance(chain_code_input_img, stored_code)
            if distance < best_distance:
                best_distance = distance
                best_match = digit
        
        return best_match
    else:
        print("No contours found in the input image")

def detect_digits_from_combined_image(image_path):
    combined_image = cv2.imread(image_path)
    height, width, _ = combined_image.shape

    num_digits = width // 100
    detected_digits = []

    for i in range(num_digits):
        digit_img = combined_image[:, i * 100:(i + 1) * 100]
        part_image_path = f'static/img/test_digits/part_{i}.png'
        if not os.path.exists('static/img/test_digits'):
            os.makedirs('static/img/test_digits')
        cv2.imwrite(part_image_path, digit_img)
        detected_digit = guess_digit_shape(part_image_path, use_skeletonize=True)
        detected_digits.append(str(detected_digit))

    return ''.join(detected_digits)

# Define the eight possible directions in Freeman Chain Code
DIRECTIONS = {
    (0, 1): 0,
    (-1, 1): 1,
    (-1, 0): 2,
    (-1, -1): 3,
    (0, -1): 4,
    (1, -1): 5,
    (1, 0): 6,
    (1, 1): 7,
}

# Function to preprocess image and find contours
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

# Function to generate Freeman Chain Code
def freeman_chain_code(contour):
    chain_code = []
    for i in range(1, len(contour)):
        delta = (contour[i][0][0] - contour[i - 1][0][0], contour[i][0][1] - contour[i - 1][0][1])
        chain_code.append(DIRECTIONS.get(delta, -1))
    return [code for code in chain_code if code != -1]

# Function to normalize chain codes for rotation invariance
def normalize_chain_code(chain_code):
    min_code = chain_code
    for i in range(len(chain_code)):
        rotated = chain_code[i:] + chain_code[:i]
        if rotated < min_code:
            min_code = rotated
    return min_code

# Function to calculate the difference between two chain codes
def chain_code_difference(code1, code2):
    min_length = min(len(code1), len(code2))
    difference = sum(1 for i in range(min_length) if code1[i] != code2[i])
    difference += abs(len(code1) - len(code2))  # Account for length differences
    return difference

# Function to extract HOG features from an image
def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (64, 64))  # Resize to a fixed size
    hog_features, _ = hog(resized_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                          block_norm='L2-Hys', visualize=True)
    return hog_features

emoji_dir = "static/img/emoji"

def train_emoji_model():
    # Membuat list untuk menyimpan fitur HOG dan label emoji
    hog_features = []
    labels = []

    # Mendapatkan daftar nama file emoji di direktori
    emoji_files = os.listdir(emoji_dir)

    # Mendefinisikan parameter HOG
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)

    # Mengambil fitur HOG dan label dari setiap gambar emoji
    for emoji_file in emoji_files:
        # Mendapatkan label emoji dari nama file (misalnya, "happy.png" menjadi "happy")
        label = os.path.splitext(emoji_file)[0]
        labels.append(label)
        
        # Membaca gambar emoji dan mengonversinya ke grayscale
        emoji_img = Image.open(os.path.join(emoji_dir, emoji_file)).convert("L")
        
        # Menghitung fitur HOG dari gambar emoji
        hog_feature = hog(emoji_img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        hog_features.append(hog_feature)

    # Mengubah list menjadi array numpy
    X = np.array(hog_features)
    y = np.array(labels)

    # Membuat pipeline untuk pemrosesan standar dan model SVM
    pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))

    # Melatih model SVM
    pipeline.fit(X, y)

    # Menyimpan model yang dilatih sebagai file .pkl
    model_path = "static/img/emoji_model.pkl"
    dump(pipeline, model_path)
    
    print("Model emoji berhasil dilatih dan disimpan sebagai", model_path)

# Fungsi untuk memuat model emoji yang telah dilatih
def load_emoji_model():
    model_path = "static/img/emoji_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        print("Model emoji belum dilatih. Silakan latih model terlebih dahulu.")
        return None

# Fungsi untuk memprediksi emoji dari gambar menggunakan model yang telah dilatih
def predict_emoji(image_path):
    model = load_emoji_model()
    if model:
        # Baca gambar dan hitung fitur HOG
        emoji_img = Image.open(image_path).convert("L")
        hog_feature = hog(emoji_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        
        # Lakukan prediksi menggunakan model
        predicted_emoji = model.predict([hog_feature])[0]
        return predicted_emoji
    else:
        return None
