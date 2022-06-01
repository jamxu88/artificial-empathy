import os
import cv2
import time
import numpy as np
from fastai.vision.all import *
from fastai.text.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

### Sources: 
### https://docs.opencv.org/4.x/
### https://elbruno.com/2019/09/25/vscode-20-lines-to-display-a-webcam-camera-feed-with-python/
### https://github.com/AnamikaAhmed/Face-Recognition-Haar-Cascade

# init camera   
execution_path = os.getcwd()

faceCascade = cv2.CascadeClassifier("./cas.xml")

learner = load_learner("./export2.pkl")


class_labels = ["Angry","Disgusted","Fearful","Happy","Neutral","Sad","Surprised"]


conf = 0.30                          # Confidence level threshold
thickness = 4                       # Thickness of rectangle 
blue = (220, 159, 0)               # Color in BGR
meanValues = (104.0, 177.0, 124.0)  # RGB mean values from ImageNet training set

camera = cv2.VideoCapture(4)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def drawRectangle(image, color, t):
    (x, y, x1, y1) = t
    cv2.rectangle(image, (x, y), (x+x1, y+y1), color, thickness)
    return image

def detectFaces(image):
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        drawRectangle(image,blue,(x,y,w,h))
        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:

                # make a prediction on the ROI
                pred = learner.predict(roi_gray)
                print(pred)
                label = pred[0]

                label_position = (x, y)
                cv2.putText(
                    frame,
                    label,
                    label_position,
                    cv2.FONT_HERSHEY_COMPLEX,
                    2,
                    (220, 159, 0),
                    3,
                )
    return image

while True:

    # Capture frame-by-frame
    ret, frame = camera.read()

    frame = detectFaces(frame)
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

