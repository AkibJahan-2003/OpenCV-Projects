import cv2
import os
import numpy as np
import face_recognition

webcam = cv2.VideoCapture("http://192.168.0.103:8080/video")

#Adding image path
pathFolder = 'Face_recognition\images'
images = []
clsName = []
myList = os.listdir(pathFolder)

for cls in myList:
    Curimg = cv2.imread(f'{pathFolder}/{cls}')
    images.append(Curimg)
    clsName.append(os.path.splitext(cls)[0])

# Encoding start from here
def findEncoding(images):
    encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodings.append(encode)
    return encodings
encodeListKnown = findEncoding(images)
print("Encoding Done")

while True:
    _ , img = webcam.read()

    imgS = cv2.resize(img,(0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeFrame = face_recognition.face_encodings(imgS)

    for encodeFace , faceLoc in zip(encodeFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = clsName[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1-35), (x2, y1), (0, 255, 0), cv2.FILLED)
            cv2.putText(img , name , (x1+6 , y1-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    img = cv2.resize(img, (640, 480))
    cv2.imshow("Face recognition", img)
    if cv2.waitKey(25) & 0xFF == 27:
        break
webcam.release()

cv2.destroyAllWindows()
