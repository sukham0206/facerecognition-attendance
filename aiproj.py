import cv2
import os
import numpy as np
import face_recognition 
from datetime import datetime, date

path = 'test'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
  curImg = cv2.imread(f'{path}/{cl}')
  images.append(curImg)
  classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(imgs):
  encodeList = []
  for img in imgs:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encodeList.append(encode)
  return encodeList

def markAttendance(name):
  x = date.today()
  with open('attendance_{}-{}-{}.csv'.format(x.day,x.month,x.year), 'w+') as f:
    dataList = f.readlines()
    nameList = []
    for line in dataList: 
      entry = line.split(',')
      nameList.append(entry[0])
    if name not in nameList:
      now = datetime.now()
      dt = now.strftime('%H:%M:%S')
      d = date.today()
      f.writelines(f'\n{name},{dt},{d}')


encodeListKnown = findEncodings(images)
print(len(encodeListKnown))


capture = cv2.VideoCapture(0)

while True:
  success, img = capture.read()
  imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
  imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

  faceLocCur = face_recognition.face_locations(imgS)
  encodeCur = face_recognition.face_encodings(imgS, faceLocCur)

  for encodeFace, faceLoc in zip(encodeCur, faceLocCur):
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
    
    matchIndex = np.argmin(faceDistance)

    if matches[matchIndex]:
      name = classNames[matchIndex].upper() 

      y1,x2,y2,x1 = faceLoc
      y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
      cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
      cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
      cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
      markAttendance(name)
      cv2.imshow('Webcam', img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

