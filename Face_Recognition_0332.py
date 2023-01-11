# FACE RECOGNITION + ATTENDANCE PROJECT | OpenCV Python | Computer Vision
# FACE RECOGNITION + ATTENDANCE PROJECT
# https://youtu.be/sz25xxF_AVE?t=1228
# Останавливаем обработку после первого найденного
# TODO https://stackoverflow.com/questions/60450364/how-to-run-dlib-face-recognition-with-gpu

import cv2
import numpy as np
import face_recognition # https://github.com/ageitgey/face_recognition#face-recognition
import os
import time
from datetime import datetime
import json

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def data_save_json(data, file):
    # path = os.path.join('', 'json_output', file)
    with open(file, 'w', encoding='utf8') as f:
        json.dump(data, f)

def faceDetection_01(imgS):
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    facesCurFrame = []

    facetracker = load_model('facetracker.h5')

    rgb = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))
    yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]
    print (type(sample_coords), f'sample_coords={sample_coords}')
    # TODO facesCurFrame.append
    sample_coords[:2]
    print(f'sample_coords = {sample_coords}')
    # print(f'sample_coords[:2] = {sample_coords[:2]}')
    # print(f'sample_coords[2:] = {sample_coords[2:]}')
    # facesCurFrame.append(tuple(sample_coords))
    facesCurFrame.append(tuple(np.multiply(sample_coords, [450, 450, 450, 450]).astype(int)))
    print(f'facesCurFrame={facesCurFrame}')
    return facesCurFrame


def findFacesOnVideo(video_file, output=False, saveToFile=False):
    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f' frame_width={frame_width} frame_height={frame_height} fps={fps}')
    video_out_file = video_file + '_out.mp4'
    if output:
        out = cv2.VideoWriter(video_out_file,cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))

    # Initialize count
    count = 0
    faces_found = []
    faces_found_first = []
    faces_names = []
    start_time = time.time()  # Время начала обработки
    # size_reduction_factor = 0.25 # 750 сек 12.5 мин 0 чел
    # size_recovery_multiplier = 4

    size_reduction_factor = 0.5  # 2151 сек 35 мин 1 чел
    size_recovery_multiplier = 2

    # size_reduction_factor = 0.33 # 1113 сек 18.5 мин 0 чел
    # size_recovery_multiplier = 3

    # size_reduction_factor = 0.7 # 3827 сек 63 мин 1 час  2 чел
    # size_recovery_multiplier = 1

    # size_reduction_factor = 1 # 7593 сек 126 мин 2.1 часа 3 чел
    # size_recovery_multiplier = 1

    # print(f'start_time={start_time}')

    while True:
        success, img = cap.read()
        count += 1
        if not success:
            break
        # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        # imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        imgS = cv2.resize(img, (0, 0), None, size_reduction_factor, size_reduction_factor)
        imgSRGB = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Face Detection:
        facesCurFrame = face_recognition.face_locations(imgSRGB)
        # TODO
        # facesCurFrame = faceDetection_01(imgS)
        # faceDetection_01(imgS)
        if len(facesCurFrame) > 0: # When on the frame exit one or more faces:
            print(f'{count} {len(facesCurFrame)} facesCurFrame={facesCurFrame}')
            for detected_face in facesCurFrame:
                y1, x2, y2, x1 = detected_face
                # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                # y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
                y1, x2, y2, x1 = y1 * size_recovery_multiplier, x2 * size_recovery_multiplier, y2 * size_recovery_multiplier, x1 * size_recovery_multiplier
                if output:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Face Recognition:
                encodesCurFrame = face_recognition.face_encodings(imgSRGB, facesCurFrame)
                frm_dic = {}
                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    # matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace, 0.5)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    # print(faceDis)
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:
                        name = classNames[matchIndex]
                        print(name)
                        # Save the frame to a file
                        if saveToFile:
                            imgFileName = video_file + '_' + name + '_' + str(count) + '.jpg'
                            cv2.imwrite(imgFileName, img)

                        y1, x2, y2, x1 = faceLoc
                        # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        # y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
                        y1, x2, y2, x1 = y1 * size_recovery_multiplier, \
                                         x2 * size_recovery_multiplier, \
                                         y2 * size_recovery_multiplier, \
                                         x1 * size_recovery_multiplier
                        if output:
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                        frm_dic['name'] = name
                        frm_dic['frame_num'] = int(count)
                        frm_dic['x1'] = int(x1)
                        frm_dic['y1'] = int(y1)
                        frm_dic['x2'] = int(x2)
                        frm_dic['y2'] = int(y2)
                        time_sec = round(int(count) / fps)
                        frm_dic['time_sec'] = time_sec
                        # print(type(frm_dic), f' frm_dic={frm_dic}')
                        faces_found.append(frm_dic)
                        if name not in faces_names:
                            faces_names.append(name)
                            faces_found_first.append(frm_dic)
                            print(f'len(faces_found_first)={len(faces_found_first)}')

        if output:
            cv2.imshow('img RGB', img)
            out.write(img)
            key = cv2.waitKey(1)
            if key == 27:
                break

        # Прерываем цикл, когда нашли первый раз
        # if len(faces_found_first) > 0:
        #     break

    run_time = time.time() - start_time


    if len(faces_found_first) > 0:
        print(type(faces_found), f'faces_found={faces_found}')
        # json_string = json.dumps(faces_found)
        json_file = video_file + '.json'
        data_save_json(faces_found, json_file)

        print(type(faces_found_first), f'faces_found_first={faces_found_first}')
        # json_string = json.dumps(faces_found_first)
        json_file = video_file + '_first' + '.json'
        data_save_json(faces_found_first, json_file)
    else:
        print('Faces not found on the video')

    print("--- %s seconds ---" % run_time)  # Время окончания обработки
    if output:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

# ------------------

# path = 'images'
path = 'images_kv'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

import dlib
# Check if dlib use CUDA?
print('dlib.DLIB_USE_CUDA=', dlib.DLIB_USE_CUDA)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('Encoding Complete')

# video_file_path = 'video/Los Puentes 2021-04-23 Day/'
# video_file_path = 'video/Tesoros de Kazan 2021-10-08 Day/'
# video_file_path = 'video/Tesoros de Kazan 2021-10-08 Evening/'
# video_file_path = 'video/Tesoros de Kazan 2021-10-09 Day/'
# video_file_path = 'video/Tesoros de Kazan 2021-10-09 Evening with Tango en Vivo/'
# video_file_path = 'video/Tesoros de Kazan 2021-10-10 Day/'
# video_file_path = 'video/Vamos A Bailar Tver tango marathon 2018-10-27 Day/'
# video_file_path = 'video/Vamos A Bailar Tver tango marathon 2018-10-26 Evening/'
# video_file_path = 'video/Vamos A Bailar Tver Tango marathon 2018-10-27 Evening/'
video_file_path = 'video/Vamos A Bailar Tver Tango marathon 2018-10-28 Day/'

# myVideoList = os.listdir(video_file_path)
myVideoList = [f for f in os.listdir(video_file_path) if f.endswith('.mp4')]
print(myVideoList)
for video_file in myVideoList:
    v = video_file_path + video_file
    print(v)
    # findFacesOnVideo(v)
    findFacesOnVideo(v, saveToFile=True)
    # findFacesOnVideo(v, output=True)

