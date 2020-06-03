import cv2
import os
from imutils.video import VideoStream
#from picamera.array import PiRGBArray
#from picamera import PiCamera
from imutils import face_utils
from keras.models import load_model
import numpy as np
import imutils
import argparse
from pygame import mixer
import time
import dlib

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mixer.init()
sound = mixer.Sound('alarm.wav')

#camera = PiCamera()
#camera.resolution = (320, 240)
#camera.framerate = 30
#camera.shutter_speed = 10000
#camera.exposure_mode = 'nightpreview'
#rawCapture = PiRGBArray(camera, size=(320, 240))


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())


lbl = ['Close', 'Open']

model = load_model('models/newcnn.h5')
path = os.getcwd()
#cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
YAWN_THRESH = 20
thicc = 2
rpred = [99]
lpred = [99]

face = cv2.CascadeClassifier('/home/pi/Desktop/DrowsinessDetection/haar_cascade_files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('/home/pi/Desktop/DrowsinessDetection/haar_cascade_files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('/home/pi/Desktop/DrowsinessDetection/haar_cascade_files/haarcascade_righteye_2splits.xml')
yawn = cv2.CascadeClassifier('/home/pi/Desktop/DrowsinessDetection/haar_cascade_files/haarcascade_frontalface_default.xml')
predictor_yawn = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


#time.sleep(0.1)
#vs = VideoStream(src=args["webcam"]).start()
vs=VideoStream(usePiCamera=True,framerate=40).start()
time.sleep(1.0)
while (True):
    
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    #frame = image.array
    #height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAYSCALE

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1,
                                      minSize=(25, 25))  # RETURNS ARRAY OF DETECTIONS WITH X,Y,W,H.
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)
    mouth = yawn.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    cv2.rectangle(frame, (0, 390), (450,450), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
        for (ex, ey, ew, eh) in mouth:
            rect = dlib.rectangle(int(ex), int(ey), int(ex + ew), int(ey + eh))

            shape = predictor_yawn(gray, rect)
            shape = face_utils.shape_to_np(shape)
            distance = lip_distance(shape)
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)
            if (distance > YAWN_THRESH):
                cv2.putText(frame, "Yawn Alert", (10, 20),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "YAWN: {:.2f}".format(distance), (50, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]  # EXTRACT RIGHT EYE FEATURES
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))  # RESIZE TO 24*24 PIXELS
        r_eye = r_eye / 255  # max color code 255 - to normalize data betweeen 0 to 1
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict_classes(r_eye)
        if (rpred[0] == 1):
            lbl = 'Open'
        if (rpred[0] == 0):
            lbl = 'Closed'
        break
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict_classes(l_eye)
        if (lpred[0] == 1):
            lbl = 'Open'
        if (lpred[0] == 0):
            lbl = 'Closed'
        break

        #if (distance > YAWN_THRESH):
         #   score += 1
          #  cv2.putText(frame, "Yawning", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if (rpred[0] == 0 and lpred[0] == 0):
        score = score + 1
        cv2.putText(frame, "Closed", (0, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # if(rpred[0]==1 or lpred[0]==1):

    else:
        score = score - 1
        cv2.putText(frame, "Open", (0, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if (score < 0):
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (60, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if (score > 15):
            # person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()

        except:  # isplaying = False
            pass
        if (thicc < 16):
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if (thicc < 2):
                thicc = 2
        cv2.rectangle(frame, (0, 0), (320, 240), (0, 0, 255), thicc)
    cv2.imshow('frame', frame)
        
    key = cv2.waitKey(1) & 0xFF
    #rawCapture.truncate(0)
    if key==ord("q"):
        break
#cap.release()
cv2.destroyAllWindows()
vs.stop()
