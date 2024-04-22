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

from skimage.morphology import skeletonize
import os

# this function is used to convert the image to grayscale
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

# this function is used to zoom in the image
def zoomin(image_source, isMatching=False):
    img = Image.open(image_source).convert("RGB")
    img_arr = np.asarray(img)
    
    '''
    Tuple new size will be assigned with these values:
    1. The new height of the image
    2. The new width of the image
    3. Number of color channel
    '''
    new_size = ((img_arr.shape[0] * 2),
                (img_arr.shape[1] * 2), img_arr.shape[2])
    new_arr = np.full(new_size, 255)
    new_arr.setflags(write=1)

    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]

    new_r = []
    new_g = []
    new_b = []

    for row in range(len(r)):
        temp_r = []
        temp_g = []
        temp_b = []
        for i in r[row]:
            temp_r.extend([i, i])
        for j in g[row]:
            temp_g.extend([j, j])
        for k in b[row]:
            temp_b.extend([k, k])
        for _ in (0, 1):
            new_r.append(temp_r)
            new_g.append(temp_g)
            new_b.append(temp_b)

    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            new_arr[i, j, 0] = new_r[i][j]
            new_arr[i, j, 1] = new_g[i][j]
            new_arr[i, j, 2] = new_b[i][j]

    new_arr = new_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)

    if (isMatching == True):
        new_img.save("static/img/matching/img_zoomin.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to zoom out the image
def zoomout(image_source, isMatching=False):
    img = Image.open(image_source).convert("RGB")
    
    x, y = img.size
    new_arr = Image.new("RGB", (int(x / 2), int(y / 2)))
    r = [0, 0, 0, 0]
    g = [0, 0, 0, 0]
    b = [0, 0, 0, 0]

    for i in range(0, int(x/2)):
        for j in range(0, int(y/2)):
            r[0], g[0], b[0] = img.getpixel((2 * i, 2 * j))
            r[1], g[1], b[1] = img.getpixel((2 * i + 1, 2 * j))
            r[2], g[2], b[2] = img.getpixel((2 * i, 2 * j + 1))
            r[3], g[3], b[3] = img.getpixel((2 * i + 1, 2 * j + 1))
            new_arr.putpixel((int(i), int(j)), (int((r[0] + r[1] + r[2] + r[3]) / 4), int(
                (g[0] + g[1] + g[2] + g[3]) / 4), int((b[0] + b[1] + b[2] + b[3]) / 4)))
    new_arr = np.uint8(new_arr)
    new_img = Image.fromarray(new_arr)
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_zoomout.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to move the image to the left
def move_left(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')
    img_arr = np.asarray(img)

    if (len(img_arr.shape) != 3):
        img_arr = np.pad(img_arr, ((0, 0), (0, 50)), 'constant')[:, 50:]
    else:
        # Color image rgb
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
        g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
        b = np.pad(b, ((0, 0), (0, 50)), 'constant')[:, 50:]
        img_arr = np.dstack((r, g, b))
    
    new_img = Image.fromarray(img_arr)
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_moveleft.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to move the image to the right
def move_right(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')
    img_arr = np.asarray(img)

    if (len(img_arr.shape) != 3):
        img_arr = np.pad(img_arr, ((0, 0), (50, 0)), 'constant')[:, :-50]
    else:
        # Color image rgb
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
        g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
        b = np.pad(b, ((0, 0), (50, 0)), 'constant')[:, :-50]
        img_arr = np.dstack((r, g, b))

    new_img = Image.fromarray(img_arr)
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_moveright.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to move the image up
def move_up(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')
    img_arr = np.asarray(img)

    if (len(img_arr.shape) != 3):
        img_arr = np.pad(img_arr, ((0, 50), (0, 0)), 'constant')[50:, :]
    else:
        # Color image rgb
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
        g = np.pad(g, ((0, 50), (0, 0)), 'constant')[50:, :]
        b = np.pad(b, ((0, 50), (0, 0)), 'constant')[50:, :]
        img_arr = np.dstack((r, g, b))

    new_img = Image.fromarray(img_arr)
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_moveup.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to move the image down
def move_down(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')
    img_arr = np.asarray(img)

    if (len(img_arr.shape) != 3):
        img_arr = np.pad(img_arr, ((50, 0), (0, 0)), 'constant')[:-50, :]
    else:
        # Color image rgb
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.pad(r, ((50, 0), (0, 0)), 'constant')[0:-50, :]
        g = np.pad(g, ((50, 0), (0, 0)), 'constant')[0:-50, :]
        b = np.pad(b, ((50, 0), (0, 0)), 'constant')[0:-50, :]
        img_arr = np.dstack((r, g, b))

    new_img = Image.fromarray(img_arr)
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_movedown.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to increase brightness to the image using addition
def brightness_addition(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')

    img_arr = np.asarray(img).astype('uint16')
    img_arr = img_arr+100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)

    if (isMatching == True):
        new_img.save("static/img/matching/img_2.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to decrease brightness to the image using substraction
def brightness_substraction(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')

    img_arr = np.asarray(img).astype('int16')
    img_arr = img_arr-100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_3.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to increase brightness to the image using multiplication
def brightness_multiplication(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')

    img_arr = np.asarray(img)
    img_arr = img_arr*1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_4.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to decrease brightness to the image using division
def brightness_division(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')

    img_arr = np.asarray(img)
    img_arr = img_arr/1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_5.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to do convolution the image with the kernel
def convolution(img, kernel):
    kernel = np.asarray(kernel)

    if len(img.shape) == 3:
        h_img, w_img, _ = img.shape
    else:
        h_img, w_img = img.shape

    out = np.zeros((h_img - 2, w_img - 2, 3), dtype=np.float32)

    for h in range(h_img - 2):
        for w in range(w_img - 2):
            # Extract the region from the image that matches the kernel size
            region = img[h:h + kernel.shape[0], w:w + kernel.shape[1], :]
            # Check if the region and kernel have compatible shapes for element-wise multiplication
            if region.shape[:2] == kernel.shape:
                S = np.multiply(region, kernel[:, :, np.newaxis])
                out[h, w] = np.sum(S, axis=(0, 1))
            else:
                # Handle the case where the shapes do not match (e.g., at image edges)
                pass

    out_ = np.clip(out, 0, 255)
    new_img = np.uint8(out_)

    return new_img

# this function is used to detect the edge of the image
def edge_detection(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')
    
    img_arr = np.asarray(img, dtype=np.float32)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_6.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to blur the image
def blur(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')

    img_arr = np.asarray(img, dtype=np.float32)
    kernel = np.array(
        [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr.astype(np.uint8))
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_7.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to sharpen the image
def sharpening(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')

    img_arr = np.asarray(img, dtype=np.float32)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr.astype(np.uint8))
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_8.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

def histogram_rgb():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path).convert('RGB')
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        #Grayscale image
        g = img_arr[:, :].flatten()
        data_g = Counter(g)
        plt.bar(list(data_g.keys()), data_g.values(), color='black')
        plt.savefig(f'static/img/grey_histogram.jpg', dpi=300)
        plt.clf()
    else:
        # Color image
        channels = ['red', 'green', 'blue']
        data_rgb = [img_arr[:, :, i].flatten() for i in range(3)]
        
        for channel, data in zip(channels, data_rgb):
            data_counter = Counter(data)
            plt.bar(list(data_counter.keys()), data_counter.values(), color=channel)
            plt.savefig(f'static/img/{channel}_histogram.jpg', dpi=300)
            plt.clf()

def df(img):  # to make a histogram (count distribution frequency)
    values = [0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i, j]] += 1
    return values

def cdf(hist):  # cumulative distribution frequency
    cdf = [0] * len(hist)  # len(hist) is 256
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1]+hist[i]
    # Now we normalize the histogram
    # What your function h was doing before
    cdf = [ele*255/cdf[-1] for ele in cdf]
    return cdf

# this function is used to equalize the histogram of the image
def histogram_equalizer(image_source, isMatching=False):
    img = cv2.imread(image_source, 0)

    my_cdf = cdf(df(img))

    # use linear interpolation of cdf to find new pixel values. Scipy alternative exists
    image_equalized = np.interp(img, range(0, 256), my_cdf)

    if (isMatching == True):
        cv2.imwrite('static/img/matching/img_9.jpg', image_equalized)
    else:
        cv2.imwrite('static/img/img_now.jpg', image_equalized)

# this function is used to threshold the image
def threshold(image_source, lower_thres, upper_thres, isMatching=False):
    img = Image.open(image_source).convert('RGB')
    img_arr = np.asarray(img)

    condition = np.logical_and(np.greater_equal(img_arr, lower_thres),
                               np.less_equal(img_arr, upper_thres))

    # make a copy of img_arr
    img_arr_copy = img_arr.copy()

    # modify the copy of img_arr
    # if it's within the inclusive range [lower_thres, upper_thres], then
    # change it's luminance to 255 (fully white)
    img_arr_copy[condition] = 255

    new_img = Image.fromarray(img_arr_copy).convert('RGB')

    if (isMatching == True):
        new_img.save("static/img/matching/img_10.jpg")
    else:
        new_img.save("static/img/img_now.jpg", format='JPEG')

# this function is used to crop the image into "number" x "number" pieces
# for puzzle game only
def crop(filename, number):
    img = Image.open(filename).convert('RGB')
    width, height = img.size
    unit_width = width // number
    unit_height = height // number

    for i in range(number):
        for j in range(number):
            left = i * unit_width
            upper = j * unit_height
            right = (i + 1) * unit_width
            lower = (j + 1) * unit_height

            img_ = img.crop((left, upper, right, lower))
            img_.save(filename[:-4] + "_crop_" + str(i + 1) + "_" + str(j + 1) + ".jpg")

# this function is used to low pass filter the image
def low_pass_filter(filename, kernel_matrix, isMatching=False):
    img = Image.open(filename).convert('RGB')
    img_arr = np.asarray(img, dtype=np.float32)

    new_arr = convolution(img_arr, kernel_matrix)
    new_img = Image.fromarray(new_arr.astype(np.uint8))
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_11.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to high pass filter the image
def high_pass_filter(filename, kernel_matrix, isMatching=False):
    img = Image.open(filename).convert('RGB')
    img_arr = np.asarray(img, dtype=np.float32)

    new_arr = convolution(img_arr, kernel_matrix)
    new_img = Image.fromarray(new_arr.astype(np.uint8))
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_12.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to band pass filter the image
def band_pass_filter(filename, kernel_matrix, isMatching=False):
    img = Image.open(filename).convert('RGB')
    img_arr = np.asarray(img, dtype=np.float32)

    new_arr = convolution(img_arr, kernel_matrix)
    new_img = Image.fromarray(new_arr.astype(np.uint8))
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_bandpassfilter.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to blur the image using cv2.GaussianBlur() method
def gaussian_blur(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')

    img_arr = np.asarray(img, dtype=np.float32)

    if img_arr.ndim == 2:
        # Grayscale image with one channel, convert it to 3 channels
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)

    # call cv2.GaussianBlur() method to blur the image
    cv_gaussianblur = cv2.GaussianBlur(img_arr, (5, 5), 0)

    new_img = Image.fromarray(cv_gaussianblur.astype(np.uint8))
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_13.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to blur the image using cv2.medianBlur() method
def median_blur(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')
    img_arr = np.asarray(img, dtype=np.float32)

    if img_arr.ndim == 2:
        # Grayscale image with one channel, convert it to 3 channels
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)

    # call cv2.medianBlur() method to blur the image
    # using the median filter (i.e. remove noise while preserving edges of image)
    cv_medianblur = cv2.medianBlur(img_arr, ksize=5)

    new_img = Image.fromarray(cv_medianblur.astype(np.uint8))
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_medianblur.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to blur the image using cv2.bilateralFilter() method
def bilateral_filter(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')

    img_arr = np.asarray(img, dtype=np.float32)

    if img_arr.ndim == 2:
        # Grayscale image with one channel, convert it to 3 channels
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)

    # call cv2.bilateralFilter() method to blur the image with bilateral filter
    # i.e. remove noise while preserving edges. We can specify the diameter of
    # each pixel neighborhood, the color sigma in the color space, and the
    # coordinate sigma in the coordinate space.
    cv_bilateralfilter = cv2.bilateralFilter(img_arr, 9, 75, 75)

    new_img = Image.fromarray(cv_bilateralfilter.astype(np.uint8))
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_14.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to blur the image using cv2.blur() method
def blur_cv(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')
    img_arr = np.asarray(img, dtype=np.float32)

    if img_arr.ndim == 2:
        # Grayscale image with one channel, convert it to 3 channels
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)

    # call cv2.blur() method to blur the image
    cv_blur = cv2.blur(img_arr, (5, 5))

    new_img = Image.fromarray(cv_blur.astype(np.uint8))

    if (isMatching == True):
        new_img.save("static/img/matching/img_blurcv.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

# this function is used to add zero padding to the image using cv2.copyMakeBorder() method
def zero_padding_cv(image_source, isMatching=False):
    img = Image.open(image_source).convert('RGB')
    img_arr = np.asarray(img, dtype=np.float32)

    if img_arr.ndim == 2:
        # Grayscale image with one channel, convert it to 3 channels
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)

    # call cv2.copyMakeBorder() method to add zero padding
    cv_zeropadding = cv2.copyMakeBorder(img_arr, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

    new_img = Image.fromarray(cv_zeropadding.astype(np.uint8))
    
    if (isMatching == True):
        new_img.save("static/img/matching/img_zeropadding.jpg")
    else:
        new_img.save("static/img/img_now.jpg")

def generate_filtered_images():
    # do 14 different filters to image based on functions in this file
    # and save the filtered images in static/img/matching folder
    grayscale("static/img/matching/img_matching.jpg", True)
    brightness_addition("static/img/matching/img_matching.jpg", True)
    brightness_substraction("static/img/matching/img_matching.jpg", True)
    brightness_multiplication("static/img/matching/img_matching.jpg", True)
    brightness_division("static/img/matching/img_matching.jpg", True)
    edge_detection("static/img/matching/img_matching.jpg", True)
    blur("static/img/matching/img_matching.jpg", True)
    sharpening("static/img/matching/img_matching.jpg", True)
    histogram_equalizer("static/img/matching/img_matching.jpg", True)
    threshold("static/img/matching/img_matching.jpg", 0, 100, True)
    low_pass_filter("static/img/matching/img_matching.jpg",
        [[0, -1, 0], [-2, 6, -2], [0, -1, 0]], True)
    high_pass_filter("static/img/matching/img_matching.jpg",
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], True)
    gaussian_blur("static/img/matching/img_matching.jpg", True)
    bilateral_filter("static/img/matching/img_matching.jpg", True)

'''
    === PRAKTIKUM MINGGU KE-9 ===
'''
def showGerigiInBinary():
    image = cv2.imread('static/img/gerigi/gerigi.jpeg')
    # change image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # save image
    gerigi_folder = "static/img/gerigi/"
    if not os.path.exists(gerigi_folder):
        os.makedirs(gerigi_folder)
    cv2.imwrite(os.path.join(gerigi_folder, 'gerigi_binary.jpg'), binary_image)

    return binary_image

def showErodedGerigiInBinary():
    binary_image = showGerigiInBinary()
    kernel = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(binary_image, kernel, iterations=1)

    # save image
    gerigi_folder = "static/img/gerigi/"
    if not os.path.exists(gerigi_folder):
        os.makedirs(gerigi_folder)
    cv2.imwrite(os.path.join(gerigi_folder, 'gerigi_eroded.jpg'), erosion)

def showDilatedGerigiInBinary():
    binary_image = showGerigiInBinary()
    kernel = np.ones((5, 5), np.uint8)

    dilation = cv2.dilate(binary_image, kernel, iterations=1)

    # save image
    gerigi_folder = "static/img/gerigi/"
    if not os.path.exists(gerigi_folder):
        os.makedirs(gerigi_folder)
    cv2.imwrite(os.path.join(gerigi_folder, 'gerigi_dilated.jpg'), dilation)

def showOpenedGerigiInBinary():
    binary_image = showGerigiInBinary()
    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # save image
    gerigi_folder = "static/img/gerigi/"
    if not os.path.exists(gerigi_folder):
        os.makedirs(gerigi_folder)
    cv2.imwrite(os.path.join(gerigi_folder, 'gerigi_opened.jpg'), opening)

def showClosedGerigiInBinary():
    binary_image = showGerigiInBinary()
    kernel = np.ones((5, 5), np.uint8)

    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # save image
    gerigi_folder = "static/img/gerigi/"
    if not os.path.exists(gerigi_folder):
        os.makedirs(gerigi_folder)
    cv2.imwrite(os.path.join(gerigi_folder, 'gerigi_closed.jpg'), closing)

def countJumlahGerigi():
    binary_image = showGerigiInBinary()

    center_x, center_y = binary_image.shape[0] // 2 - 4, binary_image.shape[1] // 2 - 4
    radius = binary_image.shape[0] // 3 + 38

    black_image = np.zeros_like(binary_image, dtype=np.uint8)
    cv2.circle(black_image, (center_x, center_y), radius, 255, -1)

    result_image = cv2.bitwise_and(binary_image, black_image)
    result_image = cv2.absdiff(binary_image, result_image)
    result_image = cv2.dilate(result_image, np.ones((5, 5), np.uint8), iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(result_image, connectivity=8)

    # save image
    gerigi_folder = "static/img/gerigi/"
    if not os.path.exists(gerigi_folder):
        os.makedirs(gerigi_folder)
    cv2.imwrite(os.path.join(gerigi_folder, 'gerigi_counted.jpg'), result_image)

    return num_labels - 1

def showBlobLinesAndDots():
    image = cv2.imread('static/img/blobLinesDots/blobLinesDots.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # save image
    blobLineDots_folder = "static/img/blobLinesDots/"
    if not os.path.exists(blobLineDots_folder):
        os.makedirs(blobLineDots_folder)
    cv2.imwrite(os.path.join(blobLineDots_folder, 'blobLinesDots_binary.jpg'), binary_image)

    return binary_image

# kernel-kernel
# Full Kernel
full_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Box Kernel
box_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Disc Kernel
disc_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

# Vertical Line Kernel
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

# Horizontal Line Kernel
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))

# Slash Kernel
slash_kernel = np.array([[0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]], dtype=np.uint8)

# Backslash Kernel
backslash_kernel = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]], dtype=np.uint8)

# Plus Kernel
plus_kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

# Cross Kernel
cross_kernel = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]], dtype=np.uint8)

# Dots kernel
dots_kernel = np.array([[0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0]], dtype=np.uint8)

def showDotsBlobOnly():
    blobImg = showBlobLinesAndDots()
    dotsImg = cv2.morphologyEx(blobImg, cv2.MORPH_OPEN, dots_kernel)

    # save image
    blobLineDots_folder = "static/img/blobLinesDots/"
    if not os.path.exists(blobLineDots_folder):
        os.makedirs(blobLineDots_folder)
    cv2.imwrite(os.path.join(blobLineDots_folder, 'blobLinesDots_dots.jpg'), dotsImg)

    return dotsImg

def showSlashBlobOnly():
    blobImg = showBlobLinesAndDots()

    slashImg = cv2.morphologyEx(blobImg, cv2.MORPH_OPEN, slash_kernel)
    slashImg = cv2.erode(slashImg, slash_kernel, iterations=3)
    slashImg = cv2.dilate(slashImg, slash_kernel, iterations=1)
    slashImg = cv2.morphologyEx(slashImg, cv2.MORPH_OPEN, slash_kernel)
    slashImg = cv2.dilate(slashImg, slash_kernel, iterations=2)

    # save image
    blobLineDots_folder = "static/img/blobLinesDots/"
    if not os.path.exists(blobLineDots_folder):
        os.makedirs(blobLineDots_folder)
    cv2.imwrite(os.path.join(blobLineDots_folder, 'blobLinesDots_slash.jpg'), slashImg)

    return slashImg

def showBackSlashBlobOnly():
    blobImg = showBlobLinesAndDots()

    backslashImg = cv2.morphologyEx(blobImg, cv2.MORPH_OPEN, disc_kernel)
    backslashImg = cv2.erode(backslashImg, backslash_kernel, iterations=3)
    backslashImg = cv2.dilate(backslashImg, backslash_kernel, iterations=3)
    backslashImg = cv2.morphologyEx(backslashImg, cv2.MORPH_OPEN, disc_kernel)

    # save image
    blobLineDots_folder = "static/img/blobLinesDots/"
    if not os.path.exists(blobLineDots_folder):
        os.makedirs(blobLineDots_folder)
    cv2.imwrite(os.path.join(blobLineDots_folder, 'blobLinesDots_backslash.jpg'), backslashImg)

    return backslashImg

def showMixedLinesBlobOnly():
    blobImg = showBlobLinesAndDots()

    mixedLinesImg = cv2.absdiff(blobImg, showSlashBlobOnly())
    mixedLinesImg = cv2.dilate(mixedLinesImg, vertical_kernel, iterations=1)
    mixedLinesImg = cv2.erode(mixedLinesImg, horizontal_kernel, iterations=3)
    mixedLinesImg = cv2.dilate(mixedLinesImg, disc_kernel, iterations=1)
    mixedLinesImg = cv2.erode(mixedLinesImg, vertical_kernel, iterations=1)
    mixedLinesImg = cv2.dilate(mixedLinesImg, horizontal_kernel, iterations=1)

    # save image
    blobLineDots_folder = "static/img/blobLinesDots/"
    if not os.path.exists(blobLineDots_folder):
        os.makedirs(blobLineDots_folder)
    cv2.imwrite(os.path.join(blobLineDots_folder, 'blobLinesDots_mixedLines.jpg'), mixedLinesImg)

    return mixedLinesImg

def generate_binary_image_for_question_3():
    imageOne = np.array([
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    ], dtype=np.uint8) * 255

    imageOneKernel = np.ones((3, 3), dtype=np.uint8)

    imageOneResized = cv2.resize(imageOne, (imageOne.shape[0] * 20, imageOne.shape[1] * 20), interpolation=cv2.INTER_NEAREST)

    # save image
    question3_folder = "static/img/question3/"
    if not os.path.exists(question3_folder):
        os.makedirs(question3_folder)
    cv2.imwrite(os.path.join(question3_folder, 'question3_imageOne.jpg'), imageOneResized)

    return imageOne, imageOneKernel

def erode_binary_image_for_question_3():
    imageOne, imageOneKernel = generate_binary_image_for_question_3()

    imageOneEroded = cv2.erode(imageOne, imageOneKernel, iterations=1)
    
    imageOneErodedResized = cv2.resize(imageOneEroded, (imageOneEroded.shape[0] * 20, imageOneEroded.shape[1] * 20), interpolation=cv2.INTER_NEAREST)

    # save image
    question3_folder = "static/img/question3/"
    if not os.path.exists(question3_folder):
        os.makedirs(question3_folder)
    cv2.imwrite(os.path.join(question3_folder, 'question3_imageOneEroded.jpg'), imageOneErodedResized)

def dilate_binary_image_for_question_3():
    imageOne, imageOneKernel = generate_binary_image_for_question_3()

    imageOneDilated = cv2.dilate(imageOne, imageOneKernel, iterations=1)
    
    imageOneDilatedResized = cv2.resize(imageOneDilated, (imageOneDilated.shape[0] * 20, imageOneDilated.shape[1] * 20), interpolation=cv2.INTER_NEAREST)

    # save image
    question3_folder = "static/img/question3/"
    if not os.path.exists(question3_folder):
        os.makedirs(question3_folder)
    cv2.imwrite(os.path.join(question3_folder, 'question3_imageOneDilated.jpg'), imageOneDilatedResized)

def open_binary_image_for_question_3():
    imageOne, imageOneKernel = generate_binary_image_for_question_3()

    imageOneOpened = cv2.morphologyEx(imageOne, cv2.MORPH_OPEN, imageOneKernel)
    
    imageOneOpenedResized = cv2.resize(imageOneOpened, (imageOneOpened.shape[0] * 20, imageOneOpened.shape[1] * 20), interpolation=cv2.INTER_NEAREST)

    # save image
    question3_folder = "static/img/question3/"
    if not os.path.exists(question3_folder):
        os.makedirs(question3_folder)
    cv2.imwrite(os.path.join(question3_folder, 'question3_imageOneOpened.jpg'), imageOneOpenedResized)

def close_binary_image_for_question_3():
    imageOne, imageOneKernel = generate_binary_image_for_question_3()

    imageOneClosed = cv2.morphologyEx(imageOne, cv2.MORPH_CLOSE, imageOneKernel)
    
    imageOneClosedResized = cv2.resize(imageOneClosed, (imageOneClosed.shape[0] * 20, imageOneClosed.shape[1] * 20), interpolation=cv2.INTER_NEAREST)

    # save image
    question3_folder = "static/img/question3/"
    if not os.path.exists(question3_folder):
        os.makedirs(question3_folder)
    cv2.imwrite(os.path.join(question3_folder, 'question3_imageOneClosed.jpg'), imageOneClosedResized)

def extract_boundary_from_binary_image_for_question_3():
    imageOne, imageOneKernel = generate_binary_image_for_question_3()

    imageOneBoundary = cv2.absdiff(imageOne, cv2.erode(imageOne, imageOneKernel, iterations=1))
    
    imageOneBoundaryResized = cv2.resize(imageOneBoundary, (imageOneBoundary.shape[0] * 20, imageOneBoundary.shape[1] * 20), interpolation=cv2.INTER_NEAREST)

    # save image
    question3_folder = "static/img/question3/"
    if not os.path.exists(question3_folder):
        os.makedirs(question3_folder)
    cv2.imwrite(os.path.join(question3_folder, 'question3_imageOneBoundary.jpg'), imageOneBoundaryResized)

'''
    === PRAKTIKUM MINGGU KE-10 ===
'''
# Function to create digit images
def create_digit_image(digit, filename):
    width, height = 100, 100
    blank_image = np.ones((height, width, 3), np.uint8) * 255

    # Determine the size and position to center the digit
    text_size = cv2.getTextSize(str(digit), cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2

    # Draw the filled digit in black
    cv2.putText(blank_image, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)

    # save every digit image within static/img/digits folder
    digits_folder = 'static/img/digits'
    if not os.path.exists(digits_folder):
        os.makedirs(digits_folder)
    
    digits_image_path = os.path.join(digits_folder, filename)

    # Save the image
    cv2.imwrite(digits_image_path, blank_image)

def create_digit_image_1_to_10():
    for digit in range(10):
        filename = f'angka_{digit}.png'
        create_digit_image(digit, filename)

def calculate_freeman_chain_code(contour):
    freeman_chain_code = ""
    directions = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]

    for i in range(1, len(contour)):
        dx = contour[i][0][0] - contour[i-1][0][0]
        dy = contour[i][0][1] - contour[i-1][0][1]

        # Temukan arah perbedaan
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

        # Draw and display the outer contours
        contour_image = np.zeros_like(img_arr, dtype=np.uint8)
        cv2.drawContours(contour_image, outer_contours, -1, (255, 255, 255), 3)
    
        return outer_contours, contour_image, contours
    else:
        print(f"No outer contours found for that digit image!")

def get_contour_from_digit(digitImgPath, use_skeletonize=False):
    gs_image = cv2.imread(f'{digitImgPath}', cv2.IMREAD_GRAYSCALE)

    # Threshold the grayscale image to create a binary image using otsu
    _, binary_image = cv2.threshold(gs_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if use_skeletonize == False:
        outer_contour, contour_image, contours = get_contours(binary_image)
        
        freeman_code = calculate_freeman_chain_code(outer_contour)
    else:
        # invert binary image
        binary_image = cv2.bitwise_not(binary_image)

        # Thin the binary image
        thinned_image = skeletonize(binary_image).astype(np.uint8) * 255

        # Find the contours on the thinned image
        contours, _ = cv2.findContours(thinned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        freeman_code = ""
        if contours:
            # display the contours
            contour_image = np.zeros_like(thinned_image, dtype=np.uint8)
            cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 3)

            # Calculate the Freeman chain code for the first contour
            freeman_code += calculate_freeman_chain_code(contours[0])
        else:
            print(f"No contours found for digit {digit}")


    return freeman_code, contour_image, contours

### METODE 1 ###
def generate_freeman_chain_code_to_env_1(digit):
    if digit == 0:
        with open('metode1.env', 'w') as f:
            f.write("")

    freeman_code_img_a, contour_image, _ = get_contour_from_digit(f'static/img/digits/angka_{digit}.png', use_skeletonize=False)

    # save freeman_code_img_a to an env file every iteration for every digit
    with open('metode1.env', 'a') as f:
        f.write(f"{digit} = {freeman_code_img_a};\n")

    # save every contour_image to a png file within static/img/contour folder
    contour_folder = 'static/img/contour'
    if not os.path.exists(contour_folder):
        os.makedirs(contour_folder)
    
    contour_image_path = os.path.join(contour_folder, f'contour_image_{digit}.png')
    # save every contour_image to a png file
    cv2.imwrite(contour_image_path, contour_image)

def generate_freeman_chain_code_to_env_1_from_1_to_10():
    for digit in range(10):
        generate_freeman_chain_code_to_env_1(digit)

### METODE 2 ###
def generate_freeman_chain_code_to_env_2(digit):
    if digit == 0:
        with open('metode2.env', 'w') as f:
            f.write("")

    freeman_code, contour_image, _ = get_contour_from_digit(f'static/img/digits/angka_{digit}.png', use_skeletonize=True)    

    # save freeman_code_img_a to an env file every iteration for every digit
    with open('metode2.env', 'a') as f:
        f.write(f"{digit} = {freeman_code};\n")
    
    # save every contour_image to a png file within static/img/contour folder
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
        # raise ValueError("Chain codes must have the same length for comparison.")
        # pad the shorter chain code with zeros
        if len(chain_code1) > len(chain_code2):
            chain_code2 = '0' * (len(chain_code1) - len(chain_code2)) + chain_code2
        else:
            chain_code1 = '0' * (len(chain_code2) - len(chain_code1)) + chain_code1

    # Initialize the distance to 0
    distance = 0

    # Iterate through the chain codes and count differences
    for symbol1, symbol2 in zip(chain_code1, chain_code2):
        if symbol1 != symbol2:
            distance += 1

    return distance

def guess_digit_shape(digitImgPath, use_skeletonize):
    chain_codes = {}
    if use_skeletonize == False:
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
        if use_skeletonize == False:
            chain_code_input_img = calculate_freeman_chain_code(contours[1])
        else:
            chain_code_input_img = calculate_freeman_chain_code(contours[0])

        best_match = None
        best_distance = float('inf')

        for digit, stored_code in chain_codes.items():
            # calculate using hamming distance
            distance = calculate_distance(chain_code_input_img, stored_code)
            if distance < best_distance:
                best_distance = distance
                best_match = digit
        
        return best_match
    else:
        print("No contours found in the input image")
    
def combine_and_detect_digits(image_paths):
    # Load all images and create a combined image
    images = [cv2.imread(image_path) for image_path in image_paths]
    combined_image = np.hstack(images)

    # Save the combined image into static/img folder
    combined_digits_folder = 'static/img'
    if not os.path.exists(combined_digits_folder):
        os.makedirs(combined_digits_folder)
    
    combined_digits_path = os.path.join(combined_digits_folder, f'combined_digits_image.png')

    cv2.imwrite(combined_digits_path, combined_image)

    # Divide the combined image into equal parts
    _, width, _ = combined_image.shape
    num_images = len(image_paths)
    images = [combined_image[:, i * (width // num_images):(i + 1) * (width // num_images)] for i in range(num_images)]

    # Detect digits for each part
    res = ""
    for i, img in enumerate(images):
        part_image_path = f'static/img/test_digits/part_{i}.png'
        if not os.path.exists('static/img/test_digits'):
            os.makedirs('static/img/test_digits')
        cv2.imwrite(part_image_path, img)
        detected_digit = guess_digit_shape(part_image_path, use_skeletonize=True)
        res += str(detected_digit)

    return res
