import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
whT = 320
confThreshold =0.5
nmsThreshold= 0.2

classesFile = "D:\Face\yolo\coco.names" #coco.names direktori
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

modelConfiguration = "D:\Face\yolo\yolov3-tiny.cfg" #file configuration direktori
modelWeights = "D:\Face\yolo\yolov3-tiny.weights" #weights direktori
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, _ = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
        cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

while True:
    _, img = cap.read()
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    print(net.getUnconnectedOutLayers())

    outputNames = [(layersNames[i[0]-1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img)
 
    cv.imshow('Image', img)
    cv.waitKey(1)
