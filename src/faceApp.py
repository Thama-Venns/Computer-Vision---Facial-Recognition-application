import numpy
import cv2 as cv
import pickle as pkl

#variable definations
header = "Ficial Recognision"
face_cascade_path = r"C:\MyProjects\FaceApp\src\cascades\data\haarcascade_frontalface_alt2.xml"
trained_data_path = r"C:\MyProjects\FaceApp\src\training.yml"
pickel_file_path = r"C:\MyProjects\FaceApp\src\labels.pickl"

#front face cascade
front_cascade = cv.CascadeClassifier(face_cascade_path)
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read(trained_data_path)

person_label = {} #Loads people's names from dataset based in ids

with open(pickel_file_path, 'rb') as file:
    id_person_label = pkl.load(file)
    person_label = {value:key for key,value in id_person_label.items()}

cp = cv.VideoCapture(0)

while(True):
    ret, frame = cp.read()

    #covert frame color to grey
    gray_color = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if frame.any():
        persons = front_cascade.detectMultiScale(gray_color, 1.5,5)
    else:
        pass
    #iterate through faces
    for(x, y, w, h) in persons:
        #print(x,y,w,h)
        roi_gray = gray_color[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #Recognixer
        id, confidence = recognizer.predict(roi_gray)

        #GUI Label vars
        name_font = cv.FONT_HERSHEY_PLAIN
        name_stroke = 2

        #Confidence conditions
        if confidence >=45 and confidence >=85:
            name = person_label[id]
            name_color = (255, 255, 255)
            print(id)
            print(person_label[id])
            cv.putText(frame, name, (x,y), name_font, 1, name_color, name_stroke, cv.LINE_AA)
        elif confidence<45:
            name = "Unknown"
            name_color = (255, 0, 0)
            cv.putText(frame, name, (x,y), name_font, 1, name_color, name_stroke, cv.LINE_AA)

        #Cascade defination
        color = (255, 50, 13)
        stroke = 2
        x_cord = x + w
        y_cord = y + h
        cv.rectangle(frame, (x,y), (x_cord, y_cord), color, stroke)

    #show frames
    cv.imshow(header, frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

#release the frame when finished
cp.release()
cv.destroyAllWindows()
