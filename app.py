import numpy as np
from PIL import Image
import image_processing # Import the image_processing.py file
import os
import random
from flask import Flask, render_template, request, make_response, session, redirect, url_for
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile
import glob

app = Flask(__name__) # Create a Flask object called "app" (__name__ is a special variable in Python)
app.secret_key = 'yEsyDNtBpySf#2023'  # Set a secret key for session management

APP_ROOT = os.path.dirname(os.path.abspath(__file__)) # Get the absolute path of the directory containing this file


def nocache(view):
    @wraps(view) # This is a decorator that updates the metadata of the view function
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)


@app.route("/index")
@app.route("/")
@nocache
def index():
    '''
        Whenever user come back to homepage "/" or "/index"
        delete every image in folders containing image
        with filename format img_*.jpg, so the app
        didn't save any image from previous image processing
        session the user had done.
    '''
    img_folders = ["static/img/matching", "static/img"]
    img_format = "img_*.jpg"
    image_files = []

    for img_folder in img_folders:
        image_files.extend(glob.glob(os.path.join(img_folder, img_format)))
    
    for image_file in image_files:
        os.remove(image_file)

    return render_template("home.html", file_path="img/image_here.jpg")

@app.route("/about")
@nocache
def about():
    return render_template('about.html')

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/upload", methods=["POST"])
@nocache
def upload():
    target = os.path.join(APP_ROOT, "static/img")
    if not os.path.isdir(target):
        if os.name == 'nt': # If the OS is Windows (nt stands for New Technology)
            os.makedirs(target)
        else: # If the OS is Linux
            os.mkdir(target)
    for file in request.files.getlist("file"):
        file.save("static/img/img_now.jpg")
    copyfile("static/img/img_now.jpg", "static/img/img_normal.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/normal", methods=["POST"])
@nocache
def normal():
    copyfile("static/img/img_normal.jpg", "static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg", dimension=False)

@app.route("/grayscale", methods=["POST"])
@nocache
def grayscale():
    image_processing.grayscale("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/zoomin", methods=["POST"])
@nocache
def zoomin():
    image_processing.zoomin("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/zoomout", methods=["POST"])
@nocache
def zoomout():
    image_processing.zoomout("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/move_left", methods=["POST"])
@nocache
def move_left():
    image_processing.move_left("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/move_right", methods=["POST"])
@nocache
def move_right():
    image_processing.move_right("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/move_up", methods=["POST"])
@nocache
def move_up():
    image_processing.move_up("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/move_down", methods=["POST"])
@nocache
def move_down():
    image_processing.move_down("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/brightness_addition", methods=["POST"])
@nocache
def brightness_addition():
    image_processing.brightness_addition("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/brightness_substraction", methods=["POST"])
@nocache
def brightness_substraction():
    image_processing.brightness_substraction("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/brightness_multiplication", methods=["POST"])
@nocache
def brightness_multiplication():
    image_processing.brightness_multiplication("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/brightness_division", methods=["POST"])
@nocache
def brightness_division():
    image_processing.brightness_division("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/histogram_equalizer", methods=["POST"])
@nocache
def histogram_equalizer():
    image_processing.histogram_equalizer("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/edge_detection", methods=["POST"])
@nocache
def edge_detection():
    image_processing.edge_detection("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/blur", methods=["POST"])
@nocache
def blur():
    image_processing.blur("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/sharpening", methods=["POST"])
@nocache
def sharpening():
    image_processing.sharpening("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/histogram_rgb", methods=["POST"])
@nocache
def histogram_rgb():
    image_processing.histogram_rgb()
    if image_processing.is_grey_scale("static/img/img_now.jpg"):
        return render_template("histogram.html", file_paths=["img/grey_histogram.jpg"])
    else:
        return render_template("histogram.html", file_paths=["img/red_histogram.jpg", "img/green_histogram.jpg", "img/blue_histogram.jpg"])

@app.route("/thresholding", methods=["POST"])
@nocache
def thresholding():
    lower_thres = int(request.form['lower_thres'])
    upper_thres = int(request.form['upper_thres'])
    image_processing.threshold("static/img/img_now.jpg", lower_thres, upper_thres)
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/puzzling_ordered", methods=["POST"])
@nocache
def puzzling_ordered():
    num_of_blocks = int(request.form['ordered_puzzle_num'])
    image_processing.crop("static/img/img_now.jpg", num_of_blocks)

    cropped_image_paths = []

    for i in range (num_of_blocks):
        for j in range (num_of_blocks):
            file_path = f"static/img/img_now_crop_{j+1}_{i+1}.jpg"
            cropped_image_paths.append(file_path)
    
    # Calculate the total width including margins
    image_width, _ = Image.open("static/img/img_now.jpg").size
    margin_width = 2   # Replace with the margin width (in pixels)

    num_images = len(cropped_image_paths)
    total_width = (image_width) + ((num_images * margin_width))  # Total width with margins
    total_width = (((image_width / num_of_blocks) + (4.2 * margin_width))) * num_of_blocks
    
    # Remove "static/" from the beginning of each file path
    cropped_image_paths = [path[7:] for path in cropped_image_paths]

    return render_template("puzzle.html", file_paths=cropped_image_paths, total_width=total_width, api_type="puzzling_ordered")

@app.route("/puzzling_unordered", methods=["POST"])
@nocache
def puzzling_unordered():
    num_of_blocks = int(request.form['unordered_puzzle_num'])
    image_processing.crop("static/img/img_now.jpg", num_of_blocks)

    cropped_image_paths = []

    for i in range (num_of_blocks):
        for j in range (num_of_blocks):
            file_path = f"static/img/img_now_crop_{j+1}_{i+1}.jpg"
            cropped_image_paths.append(file_path)
    
    # Calculate the total width including margins
    image_width, _ = Image.open("static/img/img_now.jpg").size
    margin_width = 2   # Replace with the margin width (in pixels)

    num_images = len(cropped_image_paths)
    total_width = (image_width) + ((num_images * margin_width))  # Total width with margins
    total_width = (((image_width / num_of_blocks) + (4.2 * margin_width))) * num_of_blocks
    
    # Remove "static/" from the beginning of each file path
    cropped_image_paths = [path[7:] for path in cropped_image_paths]
    # cropped_image_paths = cropped_image_paths.shuffle()

    # Create a copy of the list
    shuffled_image_paths = cropped_image_paths.copy()

    # Shuffle the copied list
    random.shuffle(shuffled_image_paths)

    return render_template("puzzle.html", file_paths=shuffled_image_paths, total_width=total_width, api_type="puzzling_unordered")

@app.route('/rgb_value_normal_per_pixel', methods=['POST'])
@nocache
def display_rgb_values():
    # Open the image
    img = Image.open('static/img/img_normal.jpg')
    width, height = img.size
    
    # Get the RGB values of each pixel
    pixels = list(img.getdata())

    # Pass the RGB values to the template
    return render_template('uploaded.html', pixels=pixels, width=width, height=height, rows=20, dimension=True, api_type="not_all")

@app.route('/rgb_value_normal_per_pixel_all', methods=['POST'])
@nocache
def display_rgb_values_all():
    # Open the image
    img = Image.open('static/img/img_normal.jpg')
    width, height = img.size
    
    # Get the RGB values of each pixel
    pixels = list(img.getdata())

    # Pass the RGB values to the template
    return render_template('uploaded.html', pixels=pixels, width=width, height=height, rows=len(pixels), dimension=True, api_type="all")


# Filtering APIs (Low-pass, High-pass, Band-pass)

@app.route("/kernel-configuration", methods=["GET", "POST"])
@nocache
def kernel_configuration():
    if request.method == "POST":
        kernel_choice = request.form.get("kernel_choice")
        filter_choice = request.form.get("filter_choice")

        if kernel_choice == "default" or kernel_choice == "custom_size" or kernel_choice == "custom_size_and_matrix":
            if kernel_choice == "default":
                # Handle the default kernel (3x3 matrix). Prepare zeros matrix
                kernel_matrix = np.zeros((3, 3))

                if filter_choice == "lowpass":
                    # Fill the kernel matrix with 1/9
                    kernel_matrix = np.full((3, 3), 1/9)
                elif filter_choice == "highpass":
                    # Fill the kernel matrix with 1
                    kernel_matrix = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
                elif filter_choice == "bandpass":
                    # Fill the kernel matrix with 1
                    kernel_matrix = np.array([[0,-1,-1],[-1,5,-1],[1,-1,0]])

            elif kernel_choice == "custom_size":
                # Handle custom size kernel
                matrix_size = int(request.form.get("kernel-size"))
                # Create an empty kernel matrix with the specified size
                kernel_matrix = np.zeros((matrix_size, matrix_size))

                # Fill randomly
                if filter_choice == "lowpass":
                    # Fill the kernel matrix with random values between 0 and 1
                    kernel_matrix = np.random.uniform(0, 1, (matrix_size, matrix_size))

                    # Normalize the kernel matrix to ensure the sum is equal to 1
                    kernel_matrix /= np.sum(kernel_matrix)

                    # check if sum of all kernel matrix elements is equal to 1
                    if not np.isclose(np.sum(kernel_matrix), 1):
                        pass
                elif filter_choice == "highpass":
                    kernel_matrix = np.full((matrix_size, matrix_size), -2)
                    
                    if (matrix_size >= 4):
                        for i in range (1, matrix_size-1):
                            for j in range (1, matrix_size-1):
                                if (
                                    (i == 1 and j == 1) or
                                    (i == 1 and j == matrix_size-2) or
                                    (i == matrix_size-2 and j == 1) or
                                    (i == matrix_size-2 and j == matrix_size-2)
                                ):
                                    kernel_matrix[i][j] = -3 * -2
                                else:
                                    pusat = matrix_size-4

                                    for k in range (2, pusat+2):
                                        if (
                                            kernel_matrix[i][k] == -2 and 
                                            ((i != 1) and (i != matrix_size-2))
                                        ):
                                            kernel_matrix[i][k] = 0

                                    kernel_matrix[i][j] *= -1
                            
                    else:
                        kernel_matrix[1][1] = 16

                elif filter_choice == "bandpass":
                    kernel_matrix = np.full((matrix_size, matrix_size), -1)
                    
                     # convert kernel_matrix so it can contain decimal values
                    kernel_matrix = kernel_matrix.astype(float)
                    
                    if (matrix_size >= 4):
                        for i in range (1, matrix_size-1):
                            for j in range (1, matrix_size-1):
                                if (
                                    (i == 1 and j == 1) or
                                    (i == 1 and j == matrix_size-2) or
                                    (i == matrix_size-2 and j == 1) or
                                    (i == matrix_size-2 and j == matrix_size-2)
                                ):
                                    kernel_matrix[i][j] = 3.3
                                else:
                                    pusat = matrix_size-4

                                    for k in range (2, pusat+2):
                                        if (
                                            kernel_matrix[i][k] == 1.4 and 
                                            ((i != 1) and (i != matrix_size-2))
                                        ):
                                            kernel_matrix[i][k] = -1.5

                                    kernel_matrix[i][j] = 1.4
                            
                    else:
                        kernel_matrix[1][1] = 9.2

            elif kernel_choice == "custom_size_and_matrix":
                # Handle custom size and matrix elements
                matrix_size = int(request.form.get("kernel-size"))
                kernel_matrix = np.zeros((matrix_size, matrix_size))
                for i in range(matrix_size):
                    for j in range(matrix_size):
                        kernel_matrix[i][j] = float(request.form.get(f"matrix[{i}][{j}]"))

            # Convert the kernel_matrix to a list before storing it in the session
            kernel_matrix_list = kernel_matrix.tolist()

            # Store the kernel_matrix and filter_choice in the session
            session['kernel_matrix'] = kernel_matrix_list
            session['filter_choice'] = filter_choice

            # Redirect to the appropriate filter endpoint based on filter_choice
            return redirect(url_for(f"{filter_choice}_filter"))
        else:
            # Handle any other cases or errors
            return "Invalid kernel choice"
        
    return render_template("kernel_config.html")

@app.route("/kernel-configuration/lowpass")
def lowpass_filter():
    # Retrieve the stored kernel matrix and filter choice from the session
    kernel_matrix = session.get('kernel_matrix')
    filter_choice = session.get('filter_choice')

    # Apply the low-pass filter with the kernel matrix
    result_image = image_processing.low_pass_filter("static/img/img_now.jpg", kernel_matrix)

    # Render the uploaded.html template with the filtered image
    return render_template("uploaded.html", file_path=result_image)

@app.route("/kernel-configuration/highpass")
def highpass_filter():
    # Retrieve the stored kernel matrix and filter choice from the session
    kernel_matrix = session.get('kernel_matrix')
    filter_choice = session.get('filter_choice')

    # Apply the high-pass filter with the kernel matrix
    result_image = image_processing.high_pass_filter("static/img/img_now.jpg", kernel_matrix)

    # Render the uploaded.html template with the filtered image
    return render_template("uploaded.html", file_path=result_image)

@app.route("/kernel-configuration/bandpass")
def bandpass_filter():
    # Retrieve the stored kernel matrix and filter choice from the session
    kernel_matrix = session.get('kernel_matrix')
    filter_choice = session.get('filter_choice')

    # Apply the band-pass filter with the kernel matrix
    result_image = image_processing.band_pass_filter("static/img/img_now.jpg", kernel_matrix)

    # Render the uploaded.html template with the filtered image
    return render_template("uploaded.html", file_path=result_image)

@app.route("/gaussian-blur", methods=["POST"])
@nocache
def gaussian_blur():
    image_processing.gaussian_blur("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/median-blur", methods=["POST"])
@nocache
def median_blur():
    image_processing.median_blur("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/bilateral-filter", methods=["POST"])
@nocache
def bilateral_filter():
    image_processing.bilateral_filter("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/blur-cv", methods=["POST"])
@nocache
def blur_cv():
    image_processing.blur_cv("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/zero-padding-cv", methods=["POST"])
@nocache
def zero_padding_cv():
    image_processing.zero_padding_cv("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/matching-game", methods=["GET"])
@nocache
def matching_game():
    return render_template("matching-home.html")

@app.route("/generate-filtered-images", methods=["GET", "POST"])
@nocache
def generate_filtered_images():
    target = os.path.join(APP_ROOT, "static/img/matching")
    if not os.path.isdir(target):
        if os.name == 'nt':
            os.makedirs(target)
        else:
            os.mkdir(target)
    for file in request.files.getlist("file"):
        file.save("static/img/matching/img_matching.jpg")
    copyfile("static/img/matching/img_matching.jpg", "static/img/matching/img_matching_normal.jpg")

    image_processing.generate_filtered_images()

    matching_image_paths = []

    for i in range (0, 14):
        file_path = f"static/img/matching/img_{i+1}.jpg"
        matching_image_paths.append(file_path)

    # Remove "static/" from the beginning of each file path
    matching_image_paths = [path[7:] for path in matching_image_paths]

    # Create a copy of the list
    shuffled_image_paths = matching_image_paths.copy()

    # Shuffle the copied list
    random.shuffle(shuffled_image_paths)

    # Double the content of the shuffled_image_paths array
    shuffled_image_paths *= 2

    # Shuffle the copied list
    random.shuffle(shuffled_image_paths)

    return render_template("matching-generated.html", file_paths=shuffled_image_paths)

@app.route("/morphological-operations-1", methods=["GET"])
@nocache
def morphological_operations_1():
    _ = image_processing.showGerigiInBinary()
    _ = image_processing.showErodedGerigiInBinary()
    _ = image_processing.showDilatedGerigiInBinary()
    _ = image_processing.showOpenedGerigiInBinary()
    _ = image_processing.showClosedGerigiInBinary()

    jumlah_gerigi = image_processing.countJumlahGerigi()

    # define variable for folder name static/img/gerigi
    folder_path = 'static/img/gerigi'

    image_files = [f for f in os.listdir(folder_path)]

    return render_template("morphological-operations-1.html", image_files=image_files, jumlah_gerigi=jumlah_gerigi)

@app.route("/morphological-operations-2", methods=["GET"])
@nocache
def morphological_operations_2():
    _ = image_processing.showBlobLinesAndDots()

    _ = image_processing.showDotsBlobOnly()
    _ = image_processing.showSlashBlobOnly()
    _ = image_processing.showBackSlashBlobOnly()
    _ = image_processing.showMixedLinesBlobOnly()

     # define variable for folder name static/img/gerigi
    folder_path = 'static/img/blobLinesDots'

    image_files = [f for f in os.listdir(folder_path)]

    return render_template("morphological-operations-2.html", image_files=image_files)

@app.route("/morphological-operations-3", methods=["GET"])
@nocache
def morphological_operations_3():
    _, _ = image_processing.generate_binary_image_for_question_3()

    image_processing.erode_binary_image_for_question_3()
    image_processing.dilate_binary_image_for_question_3()
    image_processing.open_binary_image_for_question_3()
    image_processing.close_binary_image_for_question_3()

    image_processing.extract_boundary_from_binary_image_for_question_3()

     # define variable for folder name static/img/gerigi
    folder_path = 'static/img/question3'

    image_files = [f for f in os.listdir(folder_path)]

    return render_template("morphological-operations-3.html", image_files=image_files)

@app.route("/digit-detection", methods=["GET"])
@nocache
def digit_detection():
    # populate the necessary folders with image files
    image_processing.create_digit_image_1_to_10()
    image_processing.generate_freeman_chain_code_to_env_1_from_1_to_10()
    image_processing.generate_freeman_chain_code_to_env_2_from_1_to_10()

    # hard-coded to meet the requirement
    image_paths_A = ['static/img/digits/angka_9.png']
    # image_paths_B = ['static/img/digits/angka_0.png', 'static/img/digits/angka_8.png', 'static/img/digits/angka_9.png']
    # image_paths_C = ['static/img/digits/angka_6.png', 'static/img/digits/angka_9.png', 'static/img/digits/angka_0.png', 'static/img/digits/angka_1.png']
    # image_paths_D = ['static/img/digits/angka_2.png', 'static/img/digits/angka_0.png', 'static/img/digits/angka_2.png', 'static/img/digits/angka_3.png']
    # image_paths_E = ['static/img/digits/angka_3.png', 'static/img/digits/angka_4.png']
    # image_paths_F = ['static/img/digits/angka_3.png', 'static/img/digits/angka_8.png']

    image_paths = []
    image_paths.append(image_paths_A)
    # image_paths.append(image_paths_B)
    # image_paths.append(image_paths_C)
    # image_paths.append(image_paths_D)
    # image_paths.append(image_paths_E)

    results = []
    results.append(image_processing.combine_and_detect_digits(image_paths_A))
    # results.append(image_processing.combine_and_detect_digits(image_paths_B))
    # results.append(image_processing.combine_and_detect_digits(image_paths_C))
    # results.append(image_processing.combine_and_detect_digits(image_paths_D))
    # results.append(image_processing.combine_and_detect_digits(image_paths_E))
    # results.append(image_processing.combine_and_detect_digits(image_paths_F))

    # Zip image_paths and results
    zipped_data = zip(image_paths, results)

    return render_template("digit-detection.html", zipped_data=zipped_data)

# If __name__ == '__main__' means that
# if this file is run directly, then it will run the app.run() function
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
