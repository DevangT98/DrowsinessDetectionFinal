import cv2
import os

face_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('mouth.xml')
cap = cv2.VideoCapture(0)
image_eyes = []
path = 'C:\\Users\\Devang\\PycharmProjects\\DrowsyDetect\\Generated'
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mouth = mouth_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in mouth:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        lips = mouth_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in lips:
            image_eyes.append(roi_color[ey:(ey + eh), ex:(ex + ew)])

    cv2.imshow("Face", img)
    for i, x in enumerate(image_eyes):
        cv2.imwrite(os.path.join(path, "mouth-" + str(i) + ".jpg"), x)

    if cv2.waitKey(1) & cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
