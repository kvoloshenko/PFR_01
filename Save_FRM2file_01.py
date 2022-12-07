import cv2
import os
# Save frame to file
# Сохранение фреймов с видео в отдельный файл
# video = cv2.VideoCapture(r'C:\Users\Administrator.SHAREPOINTSKY\Downloads\catty.mp4')
video = cv2.VideoCapture(r'D:\_Video_лошадки_01\костя\MVI_8783-Обрезка 04.MP4')
try:
	if not os.path.exists('foto_01'):
		os.makedirs('foto_01')
except OSError:
	print ('Error')
currentframe = 0
while(True):
	ret,frame = video.read()

	if ret:
		name = './foto_01/frame' + str(currentframe) + '.jpg'
		print ('Captured...' + name)
		cv2.imwrite(name, frame)
		currentframe += 1
	else:
		break
video.release()
cv2.destroyAllWindows()