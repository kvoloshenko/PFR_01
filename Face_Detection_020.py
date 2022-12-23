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
AUG_DATA_PATH = 'FD_02_aug_data'
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
