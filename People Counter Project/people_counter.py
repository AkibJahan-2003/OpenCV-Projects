from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *

cap = cv2.VideoCapture("../videos/people.mp4")
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

# Masking frame for clear detection
mask = cv2.imread("mask.jpg")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3 )


# For how many going up and down
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

totalCountUp = []
totalCountDown = []


while True:
    ret, frame = cap.read()
    imgRegion = cv2.bitwise_and(frame, mask)

    # Adding graphics to video
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, imgGraphics, (730, 260))
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
            if curCls == "person" and conf > 0.3:
                # cvzone.putTextRect(frame,f'{class_names[cls]} {conf}',(max(0,x1),max(35,y1)),scale = 1, thickness = 1, offset=5)
                curArray = np.array([x1,y1,x2,y2,conf])
                detection = np.vstack((detection, curArray))
    resultsTracker = tracker.update(detection)

    # This is making the lines
    cv2.line(frame , (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0,0,255), 3)
    cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 3)

    for result in resultsTracker:
        x1,y1,x2,y2,Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(frame, (x1, y1, w, h), l=15, rt=2, colorR=(255,0,0))

        cx , cy = x1+w//2, y1+h//2
        cv2.circle(frame,(cx,cy),5,(255,0,255), cv2.FILLED)

        # From here the counting start
        if limitsUp[0] <cx< limitsUp[2] and limitsUp[1]- 15 <cy< limitsUp[3]+ 15:
            if totalCountUp.count(Id) == 0:
                totalCountUp.append(Id)
        if limitsDown[0] <cx< limitsDown[2] and limitsDown[1]- 15 <cy< limitsDown[3]+ 15:
            if totalCountDown.count(Id) == 0:
                totalCountDown.append(Id)

    # Count text put on frame
    cv2.putText(frame, str(len(totalCountUp)), (929,345), cv2.FONT_HERSHEY_PLAIN, 5, (139,195,75), 7)
    cv2.putText(frame, str(len(totalCountDown)), (1200, 345), cv2.FONT_HERSHEY_PLAIN, 5, (80,83,239), 7)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()