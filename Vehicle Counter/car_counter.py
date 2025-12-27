from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *

# cap = cv2.VideoCapture('http://192.168.0.108:8080/video')
cap = cv2.VideoCapture("../videos/cars.mp4")
# cap.set(3, 640)
# cap.set(4, 480)

model = YOLO('../Yolo-Weights/yolov8n.pt')

class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

mask = cv2.imread("mask.jpg")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3 )

limits = [400, 297, 673, 297]
totalCount = []


while True:
    ret, frame = cap.read()
    imgRegion = cv2.bitwise_and(frame, mask)
    result = model(imgRegion , stream=True)

    detection = np.empty((0,5))

    for r in result:
        boxes = r.boxes
        for box in boxes:

            # Bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1, y2-y1


            # Confidence
            conf = math.ceil((box.conf[0]*100))/100

            # Class name
            cls = int(box.cls[0])
            curCls = class_names[cls]
            if curCls == "car" or curCls=="truck" or curCls=="bus" or curCls=="bicycle"  and conf > 0.3:
                # cvzone.putTextRect(frame,f'{class_names[cls]} {conf}',(max(0,x1),max(35,y1)),scale = 1, thickness = 1, offset=5)
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=15, rt=5)
                curArray = np.array([x1,y1,x2,y2,conf])
                detection = np.vstack((detection, curArray))
    resultsTracker = tracker.update(detection)
    cv2.line(frame , (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), 3)

    for result in resultsTracker:
        x1,y1,x2,y2,Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(frame, (x1, y1, w, h), l=15, rt=2, colorR=(255,0,0))
        cvzone.putTextRect(frame, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=5)

        cx , cy = x1+w//2, y1+h//2
        cv2.circle(frame,(cx,cy),5,(255,0,255), cv2.FILLED)

        if limits[0] <cx< limits[2] and limits[1]- 15 <cy< limits[3]+ 15:
            if totalCount.count(Id) == 0:
                totalCount.append(Id)

    cvzone.putTextRect(frame, f'Count: {len(totalCount)}', (50,50))

    cv2.imshow('frame', frame)
    # cv2.imshow("mask", imgRegion)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()