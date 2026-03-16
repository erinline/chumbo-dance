import cv2

stream_url = "http://192.168.0.21:81/stream"
cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame is a regular numpy array (BGR) — run your gesture detection here
    cv2.imshow("ESP32 Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
