import cv2
import pickle
import cvzone
import numpy as np

# Video Feed
cap = cv2.VideoCapture("book.mp4")

with open('libraryBookPos', 'rb') as f:
    posList = pickle.load(f)

width , height = 115, 630

def checkBookSpace(imgPro):
    spaceCounter = 0

    for pos in posList:
        x , y = pos
        imgCrop = imgPro[y:y+height, x:x+width]
        # cv2.imshow(str(x*y), imgCrop)
        count = cv2.countNonZero(imgCrop)
        cvzone.putTextRect(img, str(count),
                           (x,y+height-4),scale=1, thickness=2,
                           offset=1,colorR=(0,0,255))

        if count < 900:
            color = (0,255,0)
            thickness = 3
            spaceCounter += 1
            cvzone.putTextRect(img, str(count),
                               (x, y + height - 4), scale=1, thickness=2, offset=1, colorR=(0, 255, 0))
        else:
            color = (0,0,255)
            thickness = 2
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color,thickness)
    cvzone.putTextRect(img, f'Borrowed Books:{spaceCounter}/{len(posList)}',
                       (700, 400), scale=3, thickness=3, offset=20, colorR=(0, 200, 0))
while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.get(cv2.CAP_PROP_POS_FRAMES, 0)
    _, img = cap.read()
    img = cv2.resize(img, (1280,720))

    #Thresold images
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 0)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25,16 )
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernal = np.ones((3, 3), np.uint8)
    imgDilation = cv2.dilate(imgMedian, kernal, iterations=1)



    # create rectangle
    checkBookSpace(imgDilation)
    cv2.imshow("image", img)
    # cv2.imshow("image", imgDilation)
    if cv2.waitKey(10) & 0xFF == 27:
        break
cv2.destroyAllWindows()