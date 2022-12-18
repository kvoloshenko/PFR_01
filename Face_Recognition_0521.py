import Face_Recognition_0500 as mfr
from Face_Recognition_0500 import L1Dist
import cv2
import os
import numpy as np
import tensorflow as tf
keras = tf.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

# Reload model
siamese_model = tf.keras.models.load_model('siamesemodelv2.h5',
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# 8.2 OpenCV Real Time Verification
# https://youtu.be/LKispFFQ5GU?t=14478
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120 + 250, 200:200 + 250, :]

    cv2.imshow('Verification', frame)

    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Save input image to application_data/input_image folder
        #         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #         h, s, v = cv2.split(hsv)

        #         lim = 255 - 10
        #         v[v > lim] = 255
        #         v[v <= lim] -= 10

        #         final_hsv = cv2.merge((h, s, v))
        #         img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(os.path.join('FR_05_application_data', 'input_image', 'input_image.jpg'), frame)
        # Run verification
        results, verified = mfr.verify(siamese_model, 0.5, 0.5)
        # results, verified = verify(siamese_model, 0.9, 0.7)
        print(verified)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()