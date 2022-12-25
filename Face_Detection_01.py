# Latest 60 Fps FACE Detection | OpenCV Python | Computer Vision
#https://youtu.be/jn1HSXVmIrA
# Face_Recognition_03_Module_01
import cv2
import mediapipe as mp
import time

# video_file_path = 'video\\Los Puentes 2021-04-23 Evening\\'
video_file_path = 'video\\'

video_file_name = 'MVI_8783-Обрезка 04'
video_file_name_ext = '.MP4'
video_file = video_file_path + video_file_name + video_file_name_ext

cap = cv2.VideoCapture(video_file)
print(f'video_file={video_file}')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f' frame_width={frame_width} frame_height={frame_height} fps={fps}')

# Initialize count
count = 0
pTime = 0

start_time = time.time() #Время начала обработки
print(f'start_time={start_time}')

mpFaceDetection = mp.solutions.face_detection
mpDrow = mp.solutions.drawing_utils
# TODO
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            # TODO делать это если score > 55
            print(id, detection.score)
            print(detection.location_data.relative_bounding_box)
            # mpDrow.draw_detection(frame, detection)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bbox = int (bboxC.xmin * iw), int (bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
            # print(type(bbox), f'bbox={bbox}')
            cv2.rectangle(frame, bbox, (0, 0, 200), 2)
            cv2.putText(frame, f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

            #TODO https://youtu.be/jn1HSXVmIrA?t=983

    cTime = time.time()
    cfps = str(1 / (cTime - pTime))
    pTime = cTime
    cv2.putText(frame, f'FPS: {cfps}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    # cv2.putText(frame, cfps, (20, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    run_time = time.time() - start_time
    print("--- %s seconds ---" % run_time)  # Время окончания обработки

cap.release()
cv2.destroyAllWindows()