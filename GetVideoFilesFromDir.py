import os

video_file_path = 'video\\Los Puentes 2021-04-23 Evening\\'
myVideoList = os.listdir(video_file_path)
print(myVideoList)
for video_file in myVideoList:
    v = video_file_path + video_file
    print(v)
