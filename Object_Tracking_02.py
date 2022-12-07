import cv2
import numpy as np
from  object_detection import  ObjectDetection

# https://youtu.be/GgGro5IV-cs
# Object Tracking from scratch with OpenCV and Python

# Initialize Object Detection
od = ObjectDetection()

# cap = cv2.VideoCapture(r'D:\_Video_лошадки_01\костя\MVI_8783-Обрезка 04.MP4')
# cap = cv2.VideoCapture(r'MVI_8783-Обрезка 04.MP4')
cap = cv2.VideoCapture(r'Los Puentes 2021 part 145 vals.mp4')
# https://youtu.be/_j0OMpHlvRI


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f' frame_width={frame_width} frame_height={frame_height} fps={fps}')

# result = cv2.VideoWriter('filename.avi',
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          10, size)
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))

# Initialize count
count = 0
center_points = []

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Point current frame
    center_points_cur_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        # print(box)
        (x, y, w, h) = box
        # cx = int((x + x + w) / 2)
        # cy = int((y + y + h) / 2)
        cx = int((x ) )
        cy = int((y ) )
        center_points_cur_frame.append((cx, cy))
        center_points.append((cx, cy))

        # print("FRAME №", count, "class_ids=", class_ids, x, y, w, h)
        print("FRAME №", count, x, y, w, h)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        # TODO https://youtu.be/GgGro5IV-cs?t=1320

    # for pt in center_points:
    #     cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)



    cv2.imshow("Frame", frame)
    # cv2.waitKey(0)
    key = cv2.waitKey(1)
    # key = cv2.waitKey(0)
    if key == 27:
        break
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()