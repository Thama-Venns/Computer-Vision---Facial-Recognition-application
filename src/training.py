import os
import cv2 as cv
from PIL import Image
import numpy
import pickle as pkl

# Getting people image directory
base = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base, "data")
front_cascade = cv.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv.face.LBPHFaceRecognizer_create() #OpenCV training algorithm

#Give dataset ids and labels for Recognision
id = 1
person_id = {}

labels = []
x_train = []

#Construct data/images to train
for root, dirs, files in os.walk(img_dir):
    for file in files:
        #Check images to train from dataset
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            #Get label of file directory
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()

            if label in person_id:
                pass
            else:
                person_id[0] = "Unknown"
                person_id[label] = id
                id += 1

            id = person_id[label]
            print(person_id)

            img_pil = Image.open(path).convert("L")
            img_size = (850, 850)
            final_img = img_pil.resize(img_size, Image.ANTIALIAS)
            img_array= numpy.array(final_img, "uint8")
            #print(img_array)

            faces = front_cascade.detectMultiScale(img_array,1.5,5)

            for(x,y,w,h) in faces:
                roi = img_array[y:y+h, x:x+w]
                x_train.append(roi)
                labels.append(id)

#print(x_train)
#print(labels)
with open("people.pickl", "wb") as f:
    pkl.dump(person_id, f)

recognizer.train(x_train,numpy.array(labels))
recognizer.save("training.yml")
