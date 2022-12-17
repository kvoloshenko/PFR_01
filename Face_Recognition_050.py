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
# Make the directories
if not (os.path.exists(POS_PATH)):
    os.makedirs(POS_PATH)
    os.makedirs(NEG_PATH)
    os.makedirs(ANC_PATH)

# https://youtu.be/LKispFFQ5GU?t=1518