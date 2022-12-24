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

DATA_PATH = 'FD_02_data'
AUG_DATA_PATH = 'FD_02_aug_data'
IMAGES_PATH = os.path.join(DATA_PATH,'images')
number_images = 30

def f_plot_model(mod): # Выводим схему модели
    plot_model(mod, dpi=60, show_shapes=True)
    #model.png
    image_1 = image.load_img('model.png')
    plt.axis('off')
    # plt.imshow(image_1, interpolation='nearest')
    plt.imshow(image_1)
    plt.show()

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
# images = tf.data.Dataset.list_files(DATA_PATH + '\\images\\*.jpg')
# images.as_numpy_iterator().next()
def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img
# images = images.map(load_image)
# images.as_numpy_iterator().next()
# print (type(images))
#
# # 2.4 View Raw Images with Matplotlib
# image_generator = images.batch(4).as_numpy_iterator()
# plot_images = image_generator.next()
# fig, ax = plt.subplots(ncols=4, figsize=(15, 15))
# for idx, image in enumerate(plot_images):
#     ax[idx].imshow(image)
# plt.show()

# 3. Partition Unaugmented Data
# 3.1 MANUALLY SPLT DATA INTO TRAIN TEST AND VAL
# https://youtu.be/N_W4EYtsa10?t=2740

# 3.2 Move the Matching Labels
# https://youtu.be/N_W4EYtsa10?t=3102
# for folder in ['train', 'test', 'val']:
#     for file in os.listdir(os.path.join(DATA_PATH, folder, 'images')):
#
#         filename = file.split('.')[0] + '.json'
#         existing_filepath = os.path.join(DATA_PATH, 'labels', filename)
#         if os.path.exists(existing_filepath):
#             new_filepath = os.path.join(DATA_PATH, folder, 'labels', filename)
#             os.replace(existing_filepath, new_filepath)
# 4. Apply Image Augmentation on Images and Labels using Albumentations
# 4.1 Setup Albumentations Transform Pipeline
# https://youtu.be/N_W4EYtsa10?t=3190
import albumentations as alb
augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                       bbox_params=alb.BboxParams(format='albumentations',
                                                  label_fields=['class_labels']))
# 4.2 Load a Test Image and Annotation with OpenCV and JSON
# https://youtu.be/N_W4EYtsa10?t=3497

img = cv2.imread(os.path.join(DATA_PATH,'train', 'images', 'a5ff3cb2-8071-11ed-9e50-a8a159a2282a.jpg'))
with open(os.path.join(DATA_PATH, 'train', 'labels', 'a5ff3cb2-8071-11ed-9e50-a8a159a2282a.json'), 'r') as f:
    label = json.load(f)
print(label['shapes'][0]['points'])

# 4.3 Extract Coordinates and Rescale to Match Image Resolution
# https://youtu.be/N_W4EYtsa10?t=3682
coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]
print (type(coords))
print(f'coords={coords}')
coords = list(np.divide(coords, [640,480,640,480]))
print(f'coords={coords}')

# 4.4 Apply Augmentations and View Results
# https://youtu.be/N_W4EYtsa10?t=3808
augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
augmented['bboxes'][0][2:]
augmented['bboxes']
cv2.rectangle(augmented['image'],
              tuple(np.multiply(augmented['bboxes'][0][:2], [450,450]).astype(int)),
              tuple(np.multiply(augmented['bboxes'][0][2:], [450,450]).astype(int)),
                    (255,0,0), 2)

plt.imshow(augmented['image'])
plt.show()

# 5. Build and Run Augmentation Pipeline
# 5.1 Run Augmentation Pipeline
# https://youtu.be/N_W4EYtsa10?t=4033

# for partition in ['train','test','val']:
#     for image in os.listdir(os.path.join(DATA_PATH, partition, 'images')):
#         img = cv2.imread(os.path.join(DATA_PATH, partition, 'images', image))
#
#         coords = [0,0,0.00001,0.00001]
#         label_path = os.path.join(DATA_PATH, partition, 'labels', f'{image.split(".")[0]}.json')
#         if os.path.exists(label_path):
#             with open(label_path, 'r') as f:
#                 label = json.load(f)
#
#             coords[0] = label['shapes'][0]['points'][0][0]
#             coords[1] = label['shapes'][0]['points'][0][1]
#             coords[2] = label['shapes'][0]['points'][1][0]
#             coords[3] = label['shapes'][0]['points'][1][1]
#             coords = list(np.divide(coords, [640,480,640,480]))
#
#         try:
#             for x in range(60):
#                 augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
#                 cv2.imwrite(os.path.join(AUG_DATA_PATH, partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])
#
#                 annotation = {}
#                 annotation['image'] = image
#
#                 if os.path.exists(label_path):
#                     if len(augmented['bboxes']) == 0:
#                         annotation['bbox'] = [0,0,0,0]
#                         annotation['class'] = 0
#                     else:
#                         annotation['bbox'] = augmented['bboxes'][0]
#                         annotation['class'] = 1
#                 else:
#                     annotation['bbox'] = [0,0,0,0]
#                     annotation['class'] = 0
#
#
#                 with open(os.path.join(AUG_DATA_PATH, partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
#                     json.dump(annotation, f)
#
#         except Exception as e:
#             print(e)

# 5.2 Load Augmented Images to Tensorflow Dataset
# https://youtu.be/N_W4EYtsa10?t=4386
train_images = tf.data.Dataset.list_files(AUG_DATA_PATH + '\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
train_images = train_images.map(lambda x: x/255)
test_images = tf.data.Dataset.list_files(AUG_DATA_PATH + '\\test\\images\\*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
test_images = test_images.map(lambda x: x/255)
val_images = tf.data.Dataset.list_files(AUG_DATA_PATH + '\\val\\images\\*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
val_images = val_images.map(lambda x: x/255)
# train_images.as_numpy_iterator().next()

# 6. Prepare Labels
# 6.1 Build Label Loading Function
# https://youtu.be/N_W4EYtsa10?t=4514

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)

    return [label['class']], label['bbox']

# 6.2 Load Labels to Tensorflow Dataset
# https://youtu.be/N_W4EYtsa10?t=4595
train_labels = tf.data.Dataset.list_files(AUG_DATA_PATH + '\\train\\labels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
test_labels = tf.data.Dataset.list_files(AUG_DATA_PATH + '\\test\\labels\\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
val_labels = tf.data.Dataset.list_files(AUG_DATA_PATH + '\\val\\labels\\*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
# train_labels.as_numpy_iterator().next()

# 7. Combine Label and Image Samples
# 7.1 Check Partition Lengths
# https://youtu.be/N_W4EYtsa10?t=4747
print(f'len(train_images)={len(train_images)}')
print(f'len(train_labels)={len(train_labels)}')
print(f'len(test_images)={len(test_images)}')
print(f'len(test_labels)={len(test_labels)}')
print(f'len(val_images)={len(val_images)}')
print(f'len(val_labels)={len(val_labels)}')

# 7.2 Create Final Datasets (Images/Labels)
# https://youtu.be/N_W4EYtsa10?t=4808
train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)
test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)
val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)
# train.as_numpy_iterator().next()[1]

# 7.3 View Images and Annotations
# https://youtu.be/N_W4EYtsa10?t=4968
data_samples = train.as_numpy_iterator()
res = data_samples.next()
fig, ax = plt.subplots(ncols=4, figsize=(15, 15))
for idx in range(4):
    sample_image = res[0][idx]
    sample_coords = res[1][1][idx]

    cv2.rectangle(sample_image,
                  tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                  (255, 0, 0), 2)

    ax[idx].imshow(sample_image)
plt.show()

# 8. Build Deep Learning using the Functional API
# 8.1 Import Layers and Base Network
# https://youtu.be/N_W4EYtsa10?t=5163

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

# 8.2 Download VGG16
# https://youtu.be/N_W4EYtsa10?t=5321
vgg = VGG16(include_top=False)
print (vgg.summary())
f_plot_model(vgg) # Выводим схему модели

# 8.3 Build instance of Network
# https://youtu.be/N_W4EYtsa10?t=5524
def build_model():
    input_layer = Input(shape=(120, 120, 3))

    vgg = VGG16(include_top=False)(input_layer)

    # Classification Model
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)

    # Bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker

# 8.4 Test out Neural Network
# https://youtu.be/N_W4EYtsa10?t=5914
facetracker = build_model()
print(facetracker.summary())
f_plot_model(facetracker) # Выводим схему модели

X, y = train.as_numpy_iterator().next()
print (f'X.shape = {X.shape}')
classes, coords = facetracker.predict(X)
print(f'classes={classes}, coords={coords}')

# 9. Define Losses and Optimizers
# 9.1 Define Optimizer and LR
# https://youtu.be/N_W4EYtsa10?t=6082
print(f'len(train)={len(train)}')
batches_per_epoch = len(train)
lr_decay = (1./0.75 -1)/batches_per_epoch
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)

# 9.2 Create Localization Loss and Classification Loss
# https://youtu.be/N_W4EYtsa10?t=6174
def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - yhat[:, :2]))
    h_true = y_true[:, 3] - y_true[:, 1]
    w_true = y_true[:, 2] - y_true[:, 0]
    h_pred = yhat[:, 3] - yhat[:, 1]
    w_pred = yhat[:, 2] - yhat[:, 0]
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
    return delta_coord + delta_size

classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

# 9.3 Test out Loss Metrics
# https://youtu.be/N_W4EYtsa10?t=6297
print(f'localization_loss={localization_loss(y[1], coords)}')
print(f'classloss={classloss(y[0], classes)}')
print(f'regressloss={regressloss(y[1], coords)}')

# 10. Train Neural Network
# 10.1 Create Custom Model Class
# https://youtu.be/N_W4EYtsa10?t=6359

class FaceTracker(Model):
    def __init__(self, eyetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)

            total_loss = batch_localizationloss + 0.5 * batch_classloss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def test_step(self, batch, **kwargs):
        X, y = batch

        classes, coords = self.model(X, training=False)

        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)


model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss)


# 10.2 Train
# https://youtu.be/N_W4EYtsa10?t=6860
logdir = 'FD_02_logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])
hist = model.fit(train, epochs=40, validation_data=val, callbacks=[tensorboard_callback])

# 10.3 Plot Performance
# https://youtu.be/N_W4EYtsa10?t=7171
hist.history
fig, ax = plt.subplots(ncols=3, figsize=(20,5))

ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show()

# 11. Make Predictions
# 11.1 Make Predictions on Test Set
# https://youtu.be/N_W4EYtsa10?t=7389

test_data = test.as_numpy_iterator()
test_sample = test_data.next()
yhat = facetracker.predict(test_sample[0])
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample_image = test_sample[0][idx]
    sample_coords = yhat[1][idx]

    if yhat[0][idx] > 0.9:
        cv2.rectangle(sample_image,
                      tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                      (255, 0, 0), 2)

    ax[idx].imshow(sample_image)

# 11.2 Save the Model
# https://youtu.be/N_W4EYtsa10?t=8163

from tensorflow.keras.models import load_model
facetracker.save('facetracker.h5')
# facetracker = load_model('facetracker.h5')
# 11.3 Real Time Detection



