import os
import cv2
import time
import numpy as np

### Sources: 
### https://docs.opencv.org/4.x/
### https://elbruno.com/2019/09/25/vscode-20-lines-to-display-a-webcam-camera-feed-with-python/
### https://stackoverflow.com/questions/51842495/python-face-recognition-slow
### https://github.com/AnamikaAhmed/Face-Recognition-Haar-Cascade

# init camera
execution_path = os.getcwd()
prototxtPath = "./deploy.prototxt.txt"
caffemodelPath = "./res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxtPath, caffemodelPath)

conf = 0.30                          # Confidence level threshold
thickness = 4                       # Thickness of rectangle 
blue = (220, 159, 0)               # Color in BGR
meanValues = (104.0, 177.0, 124.0)  # RGB mean values from ImageNet training set

camera = cv2.VideoCapture(2)

def drawRectangle(image, color, t):
    (x, y, x1, y1) = t
    cv2.rectangle(image, (x, y), (x1, y1), color, thickness)
    return image

def detectFaces(image):
    h, w, _ = image.shape
    resizedImage = cv2.resize(image, (300, 300))
    blob = cv2.dnn.blobFromImage(resizedImage, 1.0, (300, 300), meanValues)

    net.setInput(blob)
    faces = net.forward()

    for i in range(0, faces.shape[2]):
        confidence = faces[0, 0, i, 2]

        if confidence > conf:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")  
            image = drawRectangle(image, blue, (x, y, x1, y1))
            print(confidence)

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

