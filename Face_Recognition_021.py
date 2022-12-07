# https://youtu.be/5yPeKQzCPdI
# https://pysource.com/2021/08/16/face-recognition-in-real-time-with-opencv-and-python/
# Installation:
#   dlib installation https://www.geeksforgeeks.org/how-to-install-dlib-library-for-python-in-windows-10/
#                     https://www.linkedin.com/pulse/installing-dlib-cuda-support-windows-10-chitransh-mundra/
#                     https://gist.github.com/nguyenhoan1988/ed92d58054b985a1b45a521fcf8fa781
#   face_recognition installation https://stackoverflow.com/questions/53737381/getting-import-error-while-importing-face-recognition-in-pycharm
import cv2
from simple_facerec import SimpleFacerec
import json
import os
import time

def data_save_json(data, file):
    # path = os.path.join('', 'json_output', file)
    with open(file, 'w', encoding='utf8') as f:
        json.dump(data, f)

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r'Los Puentes 2021 part 145 vals.mp4')
# https://youtu.be/_j0OMpHlvRI
# cap = cv2.VideoCapture(r'video\MVI_8783-Обрезка 04.MP4')
# cap = cv2.VideoCapture(r'video\Los Puentes 2021 part 145 vals.mp4')
# video_file = 'video\Los Puentes 2021-04-23 Evening\Los Puentes 2021 part 063 vals.mp4'
# video_file = 'video\Los Puentes 2021-04-23 Evening\Los Puentes 2021 part 066 milonga.mp4'

# video_file_path = 'video\\'
# video_file_name = 'MVI_8783-Обрезка 04'

video_file_path = 'video\\Los Puentes 2021-04-23 Evening\\'
# video_file_name = 'Los Puentes 2021 part 066 milonga'
# video_file_name = 'Los Puentes 2021 part 068'
# video_file_name = 'Los Puentes 2021 part 072 milonga'
# video_file_name = 'Los Puentes 2021 part 074'
video_file_name = 'Los Puentes 2021 part 077'

video_file_name_ext = '.MP4'
video_file = video_file_path + video_file_name + video_file_name_ext

cap = cv2.VideoCapture(video_file)
print(f'video_file={video_file}')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f' frame_width={frame_width} frame_height={frame_height} fps={fps}')
out = cv2.VideoWriter('video\output.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))

# Initialize count
count = 0
faces_found = []
faces_names = []

start_time = time.time() #Время начала обработки
print(f'start_time={start_time}')

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]


        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 10)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), cv2.FILLED)
        # cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        # cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 4)
        print("FRAME №", count, name, y1, x2, y2, x1)
        frm_dic = {}

        # s = "FRAME #" + str(count) + " " + name + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(x1)
        # print(s)
        # cv2.putText(frame, s, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        if name != "Unknown":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 10)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 4)
            frm_dic['name'] = name
            frm_dic['frame_num'] = int(count)
            frm_dic['x1'] = int(x1)
            frm_dic['y1'] = int(y1)
            frm_dic['x2'] = int(x2)
            frm_dic['y2'] = int(y2)
            time_sec = round(int(count) / fps)
            frm_dic['time_sec'] = time_sec
            print(type(frm_dic), f' frm_dic={frm_dic}')
            faces_found.append(frm_dic)
            faces_names.append(name)
        # else:
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 10)
        #     cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 4)



    # cv2.imshow("Frame", frame)
    #
    # key = cv2.waitKey(1)
    # if key == 27:
    #     break
    # out.write(frame)

run_time = time.time() - start_time
print("--- %s seconds ---" % run_time) #Время окончания обработки


print(type(faces_found), f'faces_found={faces_found}')
# json_string = json.dumps(faces_found)
json_file = video_file_path + video_file_name + '.json'
data_save_json(faces_found, json_file)

faces_one = set(faces_names)
print(type(faces_one), f'faces_one={faces_one}')
faces_found_first = []
for name in faces_one:
    for item in faces_found:
        if item['name'] == name:
            # time_sec = round(item['frame_num'] / fps)
            # item['time_sec'] = time_sec
            faces_found_first.append(item)
            break

print(type(faces_found_first), f'faces_found_first={faces_found_first}')
# json_string = json.dumps(faces_found_first)
json_file = video_file_path + video_file_name + '_first' + '.json'
data_save_json(faces_found_first, json_file)

cap.release()
out.release()
cv2.destroyAllWindows()