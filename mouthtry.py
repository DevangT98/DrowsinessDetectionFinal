import cv2
import numpy as np
import os

mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')

cap = cv2.VideoCapture(0)
ds_factor = 0.5
image_mouth=[]
path = 'C:\\Users\\Devang\\PycharmProjects\\DrowsyDetect\\Generated'

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    for (x,y,w,h) in mouth_rects:
        y = int(y - 0.15*h)
        image_mouth.append(cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3))
        break


    cv2.imshow('Mouth Detector', frame)
    for i, x in enumerate(image_mouth):
        cv2.imwrite(os.path.join(path, "mouth-" + str(i) + ".jpg"), x)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()