# Build a Deep Face Detection Model with Python and Tensorflow | Full Course
# https://github.com/nicknochnack/FaceDetection/blob/main/FaceDetection.ipynb
# https://youtu.be/N_W4EYtsa10

import os
import time
import uuid
import cv2
import tensorflow as tf
keras = tf.keras
import json
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image
import albumentations as alb
from tensorflow.keras.models import load_model

facetracker = load_model('facetracker.h5')
# 11.3 Real Time Detection
# https://youtu.be/N_W4EYtsa10?t=8191

# cap = cv2.VideoCapture(0)

f1='kv_photos/Los Puentes 2021 part 032.mp4_Konstantin Voloshenko_4950.jpg'
img = cv2.imread(f1)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# while cap.isOpened():
while True:
    # _, frame = cap.read()
    frame = img
    # frame = frame[50:500, 50:500, :]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.show()
    resized = tf.image.resize(rgb, (120, 120))
    plt.imshow(resized)
    plt.show()

    yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:
        # Controls the main rectangle
        cv2.rectangle(frame,
                      tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                      (255, 0, 0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame,
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                   [0, -30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                   [80, 0])),
                      (255, 0, 0), -1)

        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                                [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('EyeTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cap.release()
cv2.destroyAllWindows()