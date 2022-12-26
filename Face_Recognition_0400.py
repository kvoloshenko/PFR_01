# Распознавание лиц с InsightFace или как CatBoost имена угадывал
# Face recognition with InsightFace or how CatBoost guessed names
# https://habr.com/ru/post/699232/

import os
import pickle
from PIL import Image
import numpy as np
from typing import List
import onnxruntime as ort
from insightface.app import FaceAnalysis
from catboost import CatBoostClassifier
import shutil

def get_test_data(dir):
    # Функция get_test_data()забирает по одному изображению из каждой директории при условии,
    # что оно там не единственное.
    for subdir in os.listdir(dir):
        path = dir + subdir + '/'
        if len(os.listdir(path))>1:
            filenames = [filename for filename in os.listdir(path) ]
            face_path = path + filenames[0]
            shutil.move(face_path, 'dataset/test_faces/')

# Получение эмбеддингов
app = FaceAnalysis(name="buffalo_l",providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(256, 256))

def extract_face(filename, required_size=(256, 256)):
    print(f'extract_face, filename={filename}')
    image = Image.open(filename)
    img_arr = np.array(image)
    im = Image.fromarray((img_arr))
    im = im.resize(required_size)
    rgb_arr = np.array(im.convert('RGB'))
    emb_res = app.get(rgb_arr)
    try:
      face_array = emb_res[0].embedding
      # print(f'extract_face, face_array={face_array}')
      return face_array
    except:
      print('no embedding found for this image')

def load_face(dir):
    faces = list()
    for filename in os.listdir(dir):
        path = dir + filename
        face = extract_face(path)
        faces.append(face)
    return faces

def load_dataset(dir):
  X, y = list(), list()
  i = 1
  for subdir in os.listdir(dir):
      path = dir + subdir + '/'
      faces = load_face(path)
      labels = [subdir for i in range(len(faces))]
      print("loaded %d sample for class: %s" % (len(faces),subdir) )
      X.extend(faces)
      y.extend(labels)
      i+=1
  return np.array(X), np.array(y)


def filter_empty_embs(img_set: np.array, img_labels: List[str]):
    good_idx = [i for i,x in enumerate(img_set) if x is not None]
    clean_labels = img_labels[good_idx]
    clean_embs = img_set[good_idx]
    return clean_embs, clean_labels

get_test_data('dataset/unmasked_users/')
trainX, trainy = load_dataset('dataset/unmasked_users/')

assert len(trainX) == len(trainy)
train_emb, train_labels = filter_empty_embs(trainX, trainy)

assert len(train_emb) == len(train_labels)
print("Train_X size is {} , train_y size is {} ".format(train_emb.shape, train_labels.shape))

clf_model = CatBoostClassifier(iterations=100,
                           task_type="GPU",
                           devices='0:1')

clf_model.fit(np.array(list(train_emb)),
          train_labels,
          verbose=False)


preds = []
true_labels = []
for filename in os.listdir('dataset/test_faces/'):
    image = Image.open('dataset/test_faces/'+filename)
    img_arr = np.array(image)
    im = Image.fromarray((img_arr))
    required_size=(256, 256)
    im = im.resize(required_size)
    rgb_arr = np.array(im.convert('RGB'))
    emb_res = app.get(rgb_arr)
    try:
      face_array = emb_res[0].embedding
    except:
      print('no embedding found for this image')

    predict = clf_model.predict(face_array)
    print(type(predict), f'filename={filename} predict={predict}')

    # TODO NameError: name 'img_emb' is not defined
    # max_proba = clf_model.predict_proba(img_emb).max()
    if predict[0] in filename:
        true_labels.append(predict[0])
    else:
        true_labels.append(filename)
    preds.append(predict[0])

from sklearn.metrics import accuracy_score

print(accuracy_score(true_labels, preds))
