import numpy
import cv2 as cv

cp = cv.VideoCapture(0)

while(True):
    #capture screen frames
    ret, screen = cp.read()

    #show the frames
    cv.imshow('Facial Recognize', screen)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

#release the frame when finished
cp.release()
cv.destroyAllWindows()