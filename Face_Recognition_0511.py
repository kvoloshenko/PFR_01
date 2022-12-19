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
# anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(300)
# positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(300)
# negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(300)

anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(500)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(500)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(500)

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
# https://youtu.be/LKispFFQ5GU?t=9476
# 5.1 Setup Loss and Optimizer
# https://youtu.be/LKispFFQ5GU?t=9747
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001

# 5.2 Establish Checkpoints
# https://youtu.be/LKispFFQ5GU?t=9881
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# 5.3 Build Train Step Function
# https://youtu.be/LKispFFQ5GU?t=10080
# test_batch = train_data.as_numpy_iterator()
# batch_1 = test_batch.next()
# X = batch_1[:2]
# y = batch_1[2]
# y
@tf.function
def train_step(batch):
    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return loss
    return loss

# 5.4 Build Training Loop
# https://youtu.be/LKispFFQ5GU?t=10947
# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall


def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS + 1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # Creating a metric object
        r = Recall()
        p = Precision()

        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx + 1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

# 5.5 Train the model
# https://youtu.be/LKispFFQ5GU?t=11189
# EPOCHS = 50
EPOCHS = 100
train(train_data, EPOCHS)

# 6. Evaluate Model
# https://youtu.be/LKispFFQ5GU?t=11390
# 6.1 Import Metrics
# https://youtu.be/LKispFFQ5GU?t=11431

# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall

# 6.2 Make Predictions

# Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()
# Make predictions
y_hat = siamese_model.predict([test_input, test_val])
# Post processing the results
rez_1 = [1 if prediction > 0.5 else 0 for prediction in y_hat ]
# rez = []
# for prediction in y_hat:
#     if prediction > 0.5:
#         rez.append(1)
#     else:
#         rez.append(0)
# print(f'rez={rez}')
print(f'rez_1={rez_1}')
#
#
print(type(y_true), f'y_true={y_true}')

# 6.3 Calculate Metrics
# Creating a metric object
m = Recall()

# Calculating the recall value
m.update_state(y_true, y_hat)

# Return Recall Result
print(m.result().numpy())

# Creating a metric object
m = Precision()

# Calculating the recall value
m.update_state(y_true, y_hat)

# Return Recall Result
print(m.result().numpy())

r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true,yhat)

print(r.result().numpy(), p.result().numpy())

# 6.4 Viz Results
# https://youtu.be/LKispFFQ5GU?t=12411

# Set plot size
plt.figure(figsize=(10,8))

# Set first subplot
plt.subplot(1,2,1)
plt.imshow(test_input[0])

# Set second subplot
plt.subplot(1,2,2)
plt.imshow(test_val[0])

# Renders cleanly
plt.show()

#7. Save Model
# https://youtu.be/LKispFFQ5GU?t=12836
# Save weights
siamese_model.save('siamesemodelv2.h5')

