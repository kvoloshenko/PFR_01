
import cv2
import os
import numpy as np
import tensorflow as tf
keras = tf.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten


# Siamese L1 Distance class
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

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

# 8.1 Verification Function
# https://youtu.be/LKispFFQ5GU?t=13282
# https://youtu.be/LKispFFQ5GU?t=13734
def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('FR_05_application_data', 'verification_images')):
        input_img = preprocess(os.path.join('FR_05_application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('FR_05_application_data', 'verification_images', image))

        # Make Predictions
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection Threshold: Metric above which a prediciton is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(os.path.join('FR_05_application_data', 'verification_images')))
    verified = verification > verification_threshold

    return results, verified


# L1Dist
# Reload model
siamese_model = tf.keras.models.load_model('siamesemodelv2.h5',
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# for image in os.listdir(os.path.join('FR_05_application_data', 'verification_images')):
#     validation_img = os.path.join('FR_05_application_data', 'verification_images', image)
#     print(validation_img)

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
        results, verified = verify(siamese_model, 0.5, 0.5)
        # results, verified = verify(siamese_model, 0.9, 0.7)
        print(verified)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# https://youtu.be/LKispFFQ5GU?t=15584