import cv2
import numpy as np

cap = cv2.VideoCapture("http://192.168.0.100:8080/video")

# Color ranges in HSV (Hue, Saturation, Value)
myColor = [
    [92, 86, 0, 115, 240, 255],  # blue
    [156, 39, 0, 179, 255, 255],  # pink
    [119, 37, 11, 179, 150, 188]  # purple
]

# BGR color values for drawing
myColorValues = [
    [255, 0, 0],  # Blue
    [203, 192, 255],  # Pink
    [128, 0, 128]  # Purple
]

myPoints = []  # [x, y, colorId]


def findColor(img, myColor, myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []

    for color in myColor:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)

        if x != 0 and y != 0:
            cv2.circle(imgResult, (x, y), 10, myColorValues[count], cv2.FILLED)
            newPoints.append([x, y, count])

        count += 1
        # Optional: show masks for debugging
        # cv2.imshow(f"Mask {count}", mask)

    return newPoints


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y = 0, 0

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > 500:
            peri = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            return x + w // 2, y

    return 0, 0


def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)


# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    print("Trying alternative approach...")

    # Try with a local camera or create dummy video
    cap = cv2.VideoCapture("http://192.168.0.100:8080/video")  # Use default camera

    if not cap.isOpened():
        print("Error: Could not open any camera.")
        print("Creating a dummy video for testing...")
        # Create a dummy black image
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap = None

while True:
    if cap:
        success, img = cap.read()
        img = cv2.resize(img, (640, 480))
        if not success:
            print("Failed to grab frame")
            break
    else:
        # Create a test frame with colored circles
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 240), 30, (255, 0, 0), -1)  # Blue circle
        cv2.circle(img, (200, 150), 25, (203, 192, 255), -1)  # Pink circle
        cv2.circle(img, (440, 300), 35, (128, 0, 128), -1)  # Purple circle

    imgResult = img.copy()

    # Process the image
    newPoints = findColor(img, myColor, myColorValues)

    # Add new points to the list
    if newPoints:
        for newP in newPoints:
            myPoints.append(newP)

    # Draw all points
    if myPoints:
        drawOnCanvas(myPoints, myColorValues)

    # Display the result
    cv2.imshow("Video", imgResult)

    # Clear points when 'c' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') or key == ord('C'):
        myPoints.clear()
        print("Canvas cleared!")

    # Exit on 'q' or ESC
    if key == 27 or key == ord('q'):
        break

if cap:
    cap.release()
cv2.destroyAllWindows()