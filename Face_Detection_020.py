# https://github.com/nicknochnack/FaceDetection/blob/main/FaceDetection.ipynb
# https://youtu.be/N_W4EYtsa10

import os
import time
import uuid
import cv2
import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt

DATA_PATH = 'FD_02_data'
IMAGES_PATH = os.path.join(DATA_PATH,'images')
number_images = 30

# cap = cv2.VideoCapture(0)
# for imgnum in range(number_images):
#     print('Collecting image {}'.format(imgnum))
#     ret, frame = cap.read()
#     imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
#     cv2.imwrite(imgname, frame)
#     cv2.imshow('frame', frame)
#     time.sleep(0.5)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# 1.3 Annotate Images with LabelMe
# https://youtu.be/N_W4EYtsa10?t=1236
# https://github.com/wkentaro/labelme
# !labelme


# # Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(tf.config.list_physical_devices('GPU'))

#
# #2.3 Load Image into TF Data Pipeline
images = tf.data.Dataset.list_files(DATA_PATH + '\\images\\*.jpg')
images.as_numpy_iterator().next()
def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img
images = images.map(load_image)
images.as_numpy_iterator().next()
print (type(images))
#
# # 2.4 View Raw Images with Matplotlib
image_generator = images.batch(4).as_numpy_iterator()
plot_images = image_generator.next()
fig, ax = plt.subplots(ncols=4, figsize=(15, 15))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()

# 3. Partition Unaugmented Data
# 3.1 MANUALLY SPLT DATA INTO TRAIN TEST AND VAL
# https://youtu.be/N_W4EYtsa10?t=2740
