import cv2
import numpy as np 

def callback(x):
    pass  # 不需輸出值

# 開啟攝影機（0 通常是預設攝影機）
cap = cv2.VideoCapture(0)

cv2.namedWindow('Canny Edge Detection')
cv2.createTrackbar('L', 'Canny Edge Detection', 50, 255, callback)  # lower threshold
cv2.createTrackbar('U', 'Canny Edge Detection', 150, 255, callback)  # upper threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    l = cv2.getTrackbarPos('L', 'Canny Edge Detection')
    u = cv2.getTrackbarPos('U', 'Canny Edge Detection')

    edges = cv2.Canny(gray, l, u)

    combined = np.hstack((gray, edges))  # 並排顯示
    cv2.imshow('Canny Edge Detection', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()