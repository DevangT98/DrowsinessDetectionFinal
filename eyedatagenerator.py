import cv2
import os

face_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
image_eyes = []
path = 'C:\\Users\\Devang\\PycharmProjects\\DrowsyDetect\\Generated'
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in eyes:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eye = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eye:
            image_eyes.append(roi_color[ey:(ey + eh), ex:(ex + ew)])

    cv2.imshow("Face", img)
    for i, x in enumerate(image_eyes):
        cv2.imwrite(os.path.join(path, "mouth-" + str(i) + ".jpg"), x)

    if cv2.waitKey(1) & cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
