import cv2
import mediapipe as mp
import pyautogui
import time

x1 = y1 = x2 = y2 = 0
last_action_time = 0

# First adding a camera to show
webcam = cv2.VideoCapture("https://192.168.0.111:8080/video")

# Lightweight MediaPipe configuration
my_hand = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,           # Only 1 hand = faster
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing_utils = mp.solutions.drawing_utils

while True:
    _ , img = webcam.read()
    img = cv2.flip(img, 1)
    frame_height, frame_width , _ = img.shape

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = my_hand.process(rgb_img)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(img, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8: #Four finger
                    cv2.circle(img, (x,y), 8,(0,255,255), 2)
                    x1 = x
                    y1 = y
                if id == 4: #Thumb finger
                    cv2.circle(img, (x,y), 8,(0,0,255), 2)
                    x2 = x
                    y2 = y
                dist = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5 // 4
                cv2.line(img, (x1,y1),(x2,y2),(0,0,0), 4)
                # ✅ Limit volume press speed
                if time.time() - last_action_time > 0.3:
                    if dist > 50:
                        pyautogui.press("volumeup")
                    else:
                        pyautogui.press("volumedown")
                    last_action_time = time.time()
    cv2.imshow("Hand gesture volume control", img)
    if cv2.waitKey(25) & 0xFF == 27:
        break
webcam.release()
cv2.destroyAllWindows()

#Capturing my hand


