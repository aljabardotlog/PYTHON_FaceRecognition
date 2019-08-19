import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"name": 1}

with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi = frame[y:y + h, x:x + w]

        l_id, conf = recognizer.predict(roi_gray)
        if conf>=45:
            #print(l_id)
            #print(labels[l_id])

            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[l_id]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "aljabar_color.png"
        cv2.imwrite(img_item, roi)

        color = (255, 0, 0)
        stroke = 2
        end_x = x + w
        end_y = y + h
        cv2.rectangle(frame, (x, y), (end_x, end_y), color, stroke)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi,(ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)


    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.realease()
cv2.destroyAllWindows()