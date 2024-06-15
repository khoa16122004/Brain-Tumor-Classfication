import torch
import os
from config import *
from PIL import Image
import cv2 as cv
import numpy as np

def crop_image(image_path):
    # Đọc hình ảnh
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    original_img = img.copy()

    # finding contour and extremepoint
    blurred_img = cv.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv.threshold(blurred_img, 45, 255, cv.THRESH_BINARY)
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)
    contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)

    extLeft = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    extRight = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
    extTop = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    extBot = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

    # crop image
    cropped_img = original_img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    return cropped_img


cropped_img = crop_image(r"D:\Brain-Tumor-Classfication\Dataset\Original\training\6.jpg")
cv.imwrite("test.png", cropped_img)
# for filename in os.listdir(TRAINING_DIR):
#     file_path = os.path.join(TRAINING_DIR, filename)
#     img = cv.imread(file_path)
#     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     # resize 224 x 224 x 3
#     img = cv.resize(img, RESIZE_SHAPE)
    


    