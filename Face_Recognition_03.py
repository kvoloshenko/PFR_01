# https://youtu.be/sz25xxF_AVE

import cv2
import numpy as np
import face_recognition

imgKV = face_recognition.load_image_file('images\\Konstantin Voloshenko.jpg')
imgKVRGB = cv2.cvtColor(imgKV, cv2.COLOR_BGR2RGB)
imgKVtest = face_recognition.load_image_file('imagesTest\\kv_01.jpg')
imgKVtestRGB = cv2.cvtColor(imgKVtest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgKVRGB)[0]
encodeKV = face_recognition.face_encodings(imgKVRGB)[0]
cv2.rectangle(imgKVRGB,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgKVtestRGB)[0]
encodeTest = face_recognition.face_encodings(imgKVtestRGB)[0]
cv2.rectangle(imgKVtestRGB,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# cv2.imshow('Konstantin Voloshenko BGR', imgKV)
# cv2.waitKey(0)

results = face_recognition.compare_faces([encodeKV],encodeTest)
faceDis = face_recognition.face_distance([encodeKV],encodeTest)
print(results,faceDis)
cv2.putText(imgKVtestRGB,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Konstantin Voloshenko RGB', imgKVRGB)
cv2.waitKey(0)
cv2.imshow('Konstantin Voloshenko RGB', imgKVtestRGB )
cv2.waitKey(0)
