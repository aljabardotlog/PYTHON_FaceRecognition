import os
import numpy as np
import cv2
import pickle
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

c_id = 0
l_id = {}
y_label = []
x_train = []

for root, dirs, files in os.walk(images_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).lower()
            #print(label, path)
            if label in l_id:
                pass
            else:
                l_id[label] = c_id
                c_id += 1

            idLabel = l_id[label]
            #print(l_id)
            #y_label.append(label)
            #x_train.append(path)
            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final = pil_image.resize(size, Image.ANTIALIAS)
            image_arr = np.array(pil_image, "uint8")
            #print(image_arr)

            faces = face_cascade.detectMultiScale(image_arr, scaleFactor=1.5, minNeighbors=5)

            for (x, w, y, h) in faces:
                roi = image_arr[y:y+h, x:x+w]
                x_train.append(roi)
                y_label.append(idLabel)

#print(y_label)
#print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(l_id, f)

recognizer.train(x_train, np.array(y_label))
recognizer.save("trainer.yml")