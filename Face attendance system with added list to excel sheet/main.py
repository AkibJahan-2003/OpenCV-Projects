import cv2
import numpy as np
import face_recognition
import os
import datetime

cam = cv2.VideoCapture("http://192.168.0.115:8080/video")

pathImg = "H:/OpenCV/OpenCV projects/Face attendance system with added list to excel sheet/images"
img = []
clsName = []
myList = os.listdir(pathImg)
for cls in myList:
    curImg = cv2.imread(f'{pathImg}/{cls}')
    img.append(curImg)
    clsName.append(os.path.splitext(cls)[0])


def encoding(img):
    encode = []
    for images in img:
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(images)[0]
        encode.append(encodes)
    return encode


#Adding this in excel sheet
def markAttenance(name):
    with open('attendance.csv', 'r+',) as f:
        mydataList = f.readlines()
        nameList = []
        for line in mydataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dtString}')



encodeList = encoding(img)
print("Encoding Done")



while True:
    ret, frame = cam.read()

    imgS = cv2.resize(frame, (0, 0), None , 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    imgCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, imgCurFrame)

    for encodeLoc , imgLoc in zip(encodeCurFrame, imgCurFrame):
        matches = face_recognition.compare_faces(encodeList, encodeLoc)
        faceDis = face_recognition.face_distance(encodeList, encodeLoc)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = clsName[matchIndex].upper()
            y1, x2, y2, x1 = imgLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(frame, (x1, y1-35), (x2, y1), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            markAttenance(name)


    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
