# https://youtu.be/5yPeKQzCPdI
# https://pysource.com/2021/08/16/face-recognition-in-real-time-with-opencv-and-python/
# Installation:
#   dlib installation https://www.geeksforgeeks.org/how-to-install-dlib-library-for-python-in-windows-10/
#   face_recognition installation https://stackoverflow.com/questions/53737381/getting-import-error-while-importing-face-recognition-in-pycharm

import cv2
import face_recognition

f1='foto_01/frame1086'
f1='foto_01/frame1225'

img = cv2.imread("Messi1.webp")
# img = cv2.imread(f1)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]
print(type(img_encoding), f'img_encoding = {img_encoding}')

img2 = cv2.imread("images/Messi_01.jpg")
# img2 = cv2.imread("images/Elon Musk.jpg")
# img2 = cv2.imread(f2)
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]


result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

cv2.imshow("Img", img)
cv2.imshow("Img 2", img2)
cv2.waitKey(0)