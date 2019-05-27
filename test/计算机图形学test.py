import face_recognition as fr
import cv2
import os
import time
import itertools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import multiprocessing


class get_file(object):
 """docstring for get_file"""
 def __init__(self, file_dir):
  super(get_file, self).__init__()
  self.file_dir = file_dir

 def func(self):
  image_rule = ['png', 'jpg', 'jpeg']
  labels = []
  images = []
  temp = []
  for root, sub_folders, files in os.walk(self.file_dir):
   for name in files:
    if name.split('.')[-1] in image_rule:
     images.append(os.path.join(root, name))
   for name in sub_folders:
    temp.append(os.path.join(root, name))
  i = 0
  for one_folder in temp:
   label = one_folder.split('/')[-1]
   labels = np.append(labels, label)
   i += 1
  temp = np.array([images, labels])
  image_list = list(temp[0])
  label_list = list(temp[1])
  print(image_list, label_list)
  return image_list, label_list


def thread(id, label_list, known_faces):
 # for id in index_th:
 video_catch = cv2.VideoCapture(id)
 # video_catch.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280);
 # video_catch.set(cv2.CAP_PROP_FRAME_WIDTH, 960);
 # video_catch.set(cv2.CAP_PROP_FPS, 30.0);


 face_locations = []
 face_encodings = []
 face_names = []

 print('%s isOpened!'%id, video_catch.isOpened())
 if not video_catch.isOpened():
  return

 passfps = 0
 while True:
  ret, frame = video_catch.read()
  # print('%s' % id, ret)

  # passfps += 1
  # if passfps % 2 != 0:
  #  continue
  # passftp = 0



  if not ret:
   break
  # print(frame.shape)
  frameface = frame[100:600, 300:900, :]
  cv2.imwrite('test.png', frameface)
  small_frame = cv2.resize(frameface, (0, 0), fx=0.5, fy=0.5)
  # print(small_frame.shape)
  rgb_frame = small_frame[:, :, ::-1]
  

  face_locations = fr.face_locations(rgb_frame, model='cnn')  # ,model="cnn"
  # print(len(face_locations))
  face_encodings = fr.face_encodings(rgb_frame, face_locations)

  face_names = []

  for face_encoding in face_encodings:
   match = fr.compare_faces(known_faces, face_encoding, tolerance=0.425) # 0.425

   name = "???"
   for i in range(len(label_list)):
    if match[i]:
     name = label_list[i]

   face_names.append(name)

  for (top, right, bottom, left), name in zip(face_locations, face_names):

   if not name:
    continue
   top *= 2
   right *= 2
   bottom *= 2
   left *= 2
   top += 100
   bottom += 100
   left += 300
   right += 300
   pil_im = Image.fromarray(frame)
   draw = ImageDraw.Draw(pil_im)
   font = ImageFont.truetype('./STHeiti Medium.ttc', 24,
           encoding='utf-8')
   draw.text((left + 6, bottom - 25), name, (0, 0, 255), font=font)

   frame = np.array(pil_im)
   cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
  cv2.imshow('Video_%s' % id, frame)

  c = cv2.waitKey(1)
  if c & 0xFF == ord("q"):
   break

 video_catch.release()
 cv2.destroyWindow('Video_%s' % id)


if __name__ == '__main__':

 path = './database/face_recognition_database'
 face_database = get_file(path)
 image_list, label_list = face_database.func()

 known_faces = []
 for i in range(len(image_list)):
  tmp = fr.load_image_file(image_list[i])
  known_faces.append(list(fr.face_encodings(tmp)[0]))

 # num_camera = num_cam().func()
 num_cam_list = []
 # for i in range(num_camera):
 #  num_cam_list.append(i)
 # num_cam_list.append('rtsp://admin:ddxxzx123@10.42.25.119:554/Streaming/Channels/1')
 num_cam_list.append('./Scene1Morning1.avi')
 # /home/jonty/Documents/trash/Scene1Morning1.avi

 print(num_cam_list)

 thread('./Scene1Morning1.avi', label_list, known_faces)