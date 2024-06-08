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
from flask import Flask, render_template, Response
import cv2
from face_swap import get_landmarks, apply_delaunay_triangulation
import numpy as np
import os
from PIL import Image

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

@app.route("/binary", methods=["POST"])
@nocache
def binary():
    image_processing.binary("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/dilasi", methods=["POST"])
@nocache
def dilasi():
    image_processing.dilasi("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/erosi", methods=["POST"])
@nocache
def erosi():
    image_processing.erosi("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/opening", methods=["POST"])
@nocache
def opening():
    image_processing.opening("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/closing", methods=["POST"])
@nocache
def closing():
    image_processing.closing("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")

@app.route("/countObject", methods=["POST"])
@nocache
def countObject():
    image_processing.countObject("static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route ("/digit-detection", methods=["GET"])
@nocache
def digit_detection():
    # Generate Freeman chain codes for digits 0-9
    image_processing.generate_freeman_chain_code_to_env_1_from_1_to_10()
    image_processing.generate_freeman_chain_code_to_env_2_from_1_to_10()

    # Create a combined image of digits
    combined_image_path = image_processing.create_combined_digit_image("10101909823", "combined_image")

    # Detect digits from the combined image
    detected_digits = image_processing.detect_digits_from_combined_image(combined_image_path)

    # Prepare data for rendering
    result = {
        "image_path": combined_image_path.replace(os.sep, '/'),  # Replace backslashes with forward slashes
        "detected_digits": detected_digits
    }

    return render_template("digit-detection.html", result=result)

@app.route("/emoji-detection", methods=["GET"])
@nocache
def emoji_detection():
    # Path to the folder containing emoji images
    emoji_folder = "static/img/emoji"

    # Generate paths for all images in the emoji directory
    emoji_paths = {file.split('.')[0]: os.path.join(emoji_folder, file).replace('\\', '/')
                   for file in os.listdir(emoji_folder) if file.endswith('.png')}

    # Predict and display expressions for all emojis
    predictions = {}
    for name, path in emoji_paths.items():
        emoji_img = Image.open(path)
        # Assume you have defined image_processing.predict_emoji function
        hog_features = image_processing.extract_hog_features(path)
        prediction = image_processing.predict_emoji(path)  # Fix the parameter to path
        # Store image path in predictions
        predictions[name] = {"image": emoji_img, "prediction": prediction, "image_path": path}

    return render_template("emoji-detection.html", predictions=predictions)

<<<<<<< HEAD

app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Global variables for the uploaded image and landmarks
uploaded_image = None
uploaded_landmarks = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    global uploaded_image, uploaded_landmarks
    file = request.files['file']
    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        uploaded_image = cv2.imread(image_path)
        uploaded_landmarks = get_landmarks(uploaded_image)
        if uploaded_landmarks is None:
            return "Error: Tidak dapat mendeteksi landmark pada gambar yang diupload.", 400
        return redirect(url_for('video_stream'))
    return "No file uploaded", 400

# Video capture from webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    global uploaded_image, uploaded_landmarks
    while True:
        success, frame = cap.read()
        if not success:
            break

        landmarks2 = get_landmarks(frame)
        if landmarks2 is not None and uploaded_landmarks is not None:
            frame = apply_delaunay_triangulation(uploaded_image, frame, uploaded_landmarks, landmarks2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_stream')
def video_stream():
    return render_template('video_stream.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

=======
>>>>>>> 7a1080a1ffdf6a77ff87cba5540a9c9aef3c897a
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")