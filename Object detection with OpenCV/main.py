import cv2

thres = 0.5
# img = cv2.imread('bill.jpg')
cap = cv2.VideoCapture('http://192.168.0.102:8080/video')

cap.set(3, 640)
cap.set(4, 480)



clsName = []
clsFile = 'coco.names'
with open(clsFile, 'rt') as f:
    clsName = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    _ , img = cap.read()
    img = cv2.resize(img, (640, 480))
    clsIds , confs , bbox = net.detect(img, confThreshold=thres)
    print(clsIds , bbox)
    if len(clsIds) != 0:
        for clsId , confidence ,box in zip(clsIds.flatten() , confs.flatten() , bbox):
            cv2.rectangle(img, box, color=(0,255,0),thickness=2)
            cv2.putText(img , clsName[clsId-1].upper(), (box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(img,str(round(confidence*100,2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)




    cv2.imshow('IMAGE', img)
    if cv2.waitKey(25) & 0xFF == 27:
        break
cv2.destroyAllWindows()
