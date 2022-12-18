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
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt

def f_plot_model(mod): # Выводим схему модели
    plot_model(mod, dpi=60, show_shapes=True)
    #model.png
    image_1 = image.load_img('model.png')
    plt.axis('off')
    # plt.imshow(image_1, interpolation='nearest')
    plt.imshow(image_1)
    plt.show()

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
# anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(3000)
# positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(3000)
# negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(3000)
anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(300)
# dir_test = anchor.as_numpy_iterator()
# print(dir_test.next())

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


# img = preprocess(DATA_PATH+'\\anchor\\aaa48b8c-7e22-11ed-b244-a8a159a2282a.jpg')
# print(img.numpy().max())
# plt.imshow(img)
# plt.show()


# dataset.map(preprocess)

# https://youtu.be/LKispFFQ5GU?t=5108
# 3.3 Create Labelled Dataset
# (anchor, positive) => 1,1,1,1,1
# (anchor, negative) => 0,0,0,0,0
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)
samples = data.as_numpy_iterator()
exampple = samples.next()
print(exampple)

# https://youtu.be/LKispFFQ5GU?t=5592
# 3.4 Build Train and Test Partition
def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

res = preprocess_twin(*exampple)
plt.imshow(res[1])
plt.show()
print(res[2])

# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)
# data = data.shuffle(buffer_size=10000)
# Training partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)
# Testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# https://youtu.be/LKispFFQ5GU?t=6674
# 4. Model Engineering
# 4.1 Build Embedding Layer
# inp = Input(shape=(100,100,3), name='input_image')
# c1 = Conv2D(64, (10,10), activation='relu')(inp)
# m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
# c2 = Conv2D(128, (7,7), activation='relu')(m1)
# m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
# c3 = Conv2D(128, (4,4), activation='relu')(m2)
# m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
# c4 = Conv2D(256, (4,4), activation='relu')(m3)
# f1 = Flatten()(c4)
# d1 = Dense(4096, activation='sigmoid')(f1)
# mod = Model(inputs=[inp], outputs=[d1], name='embedding')
# print(mod.summary())
# f_plot_model(mod) # Выводим схему модели


def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')

embedding = make_embedding()
print(embedding.summary())
f_plot_model(embedding) # Выводим схему модели

# 4.2 Build Distance Layer
# https://youtu.be/LKispFFQ5GU?t=8442

# Siamese L1 Distance class
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# l1 = L1Dist()
# l1(anchor_embedding, validation_embedding)

# 4.3 Make Siamese Model
# input_image = Input(name='input_img', shape=(100,100,3))
# validation_image = Input(name='validation_img', shape=(100,100,3))
# inp_embedding = embedding(input_image)
# val_embedding = embedding(validation_image)
# siamese_layer = L1Dist()
# distances = siamese_layer(inp_embedding, val_embedding)
# classifier = Dense(1, activation='sigmoid')(distances)
# classifier
# siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
#
# print(siamese_network.summary())
# f_plot_model(siamese_network) # Выводим схему модели


def make_siamese_model():
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()
print(siamese_model.summary())
f_plot_model(siamese_model) # Выводим схему модели

# 5. Training
# 5.1 Setup Loss and Optimizer
