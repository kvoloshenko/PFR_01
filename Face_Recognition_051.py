#https://www.youtube.com/watch?v=LKispFFQ5GU
# https://github.com/nicknochnack/FaceRecognition/blob/main/Facial%20Verification%20with%20a%20Siamese%20Network%20-%20Final.ipynb

# Import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
# Import tensorflow dependencies - Functional API
import tensorflow as tf
keras = tf.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
print (type(gpus), f'gpus ={gpus}')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print (type(gpu), f'gpu ={gpu}')

# Setup paths
DATA_PATH = 'FR_05_data'
POS_PATH = os.path.join(DATA_PATH, 'positive')
NEG_PATH = os.path.join(DATA_PATH, 'negative')
ANC_PATH = os.path.join(DATA_PATH, 'anchor')

# https://youtu.be/LKispFFQ5GU?t=4120
#
# 3. Load and Preprocess Images
# 3.1 Get Image Directories
anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(3000)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(3000)
dir_test = anchor.as_numpy_iterator()
print(dir_test.next())

# 3.2 Preprocessing - Scale and Resize
def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100, 100))
    # Scale image to be between 0 and 1
    img = img / 255.0

    # Return image
    return img


img = preprocess(DATA_PATH+'\\anchor\\aaa48b8c-7e22-11ed-b244-a8a159a2282a.jpg')
print(img.numpy().max())
plt.imshow(img)
plt.show()


# dataset.map(preprocess)