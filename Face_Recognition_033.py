# FACE RECOGNITION + ATTENDANCE PROJECT
# https://youtu.be/sz25xxF_AVE?t=1228
# Останавливаем обработку после первого найденного

import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import datetime
import json

# path = 'images'
path = 'images_kv'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

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

encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('Encoding Complete')

# video_file_path = 'video\\'
# video_file_name = 'MVI_8783-Обрезка 04'
video_file_path = 'video\\Los Puentes 2021-04-23 Evening\\'
# video_file_name = 'Los Puentes 2021 part 066 milonga_fps_25_res_360'
# video_file_name = 'Los Puentes 2021 part 066 milonga_fps_25_res_720'
video_file_name = 'Los Puentes 2021 part 066 milonga'

video_file_name_ext = '.MP4'
video_file = video_file_path + video_file_name + video_file_name_ext
print(f'video_file={video_file}')
cap = cv2.VideoCapture(video_file)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f' frame_width={frame_width} frame_height={frame_height} fps={fps}')
video_out_file = video_file_path + video_file_name + '_out' + video_file_name_ext
# out = cv2.VideoWriter(video_out_file,cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))

# Initialize count
count = 0
faces_found = []
faces_found_first = []
faces_names = []
start_time = time.time() #Время начала обработки
# print(f'start_time={start_time}')

while True:
    success, img  = cap.read()
    count += 1
    if not success:
        break
    # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    imgSRGB = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSRGB)
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

            y1, x2, y2, x1 = faceLoc
            # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
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

    # cv2.imshow('img RGB', img)
    # out.write(img)
    # key = cv2.waitKey(1)
    # if key == 27:
    #     break

    # Прерываем цикл, когда нашли первый раз
    if len(faces_found_first) > 0:
        break

run_time = time.time() - start_time
print("--- %s seconds ---" % run_time) #Время окончания обработки

print(type(faces_found), f'faces_found={faces_found}')
# json_string = json.dumps(faces_found)
json_file = video_file_path + video_file_name + '.json'
data_save_json(faces_found, json_file)


print(type(faces_found_first), f'faces_found_first={faces_found_first}')
# json_string = json.dumps(faces_found_first)
json_file = video_file_path + video_file_name + '_first' + '.json'
data_save_json(faces_found_first, json_file)

cap.release()
# out.release()
cv2.destroyAllWindows()