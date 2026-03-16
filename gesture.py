import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import socket
import sys

# ---------- CONFIG ----------
stream_url = "http://192.168.0.21:81/stream"
CHUMBO_IP = "192.168.0.21"
UDP_PORT = 5000
# ----------------------------

cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("❌ Failed to open video source.")
    sys.exit(1)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- MediaPipe Tasks setup ---
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="pose_landmarker_full.task"),
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# Connection constants (replaces legacy mp.solutions references)
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
])
POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
])

# Landmark index for wrist in the hand model (same index as before)
WRIST = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame grab failed.")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pose_result = pose_landmarker.detect(mp_image)
    hand_result = hand_landmarker.detect(mp_image)

    # --- Draw pose ---
    if pose_result.pose_landmarks:
        for landmarks in pose_result.pose_landmarks:
            for connection in POSE_CONNECTIONS:
                a, b = connection
                ax = int(landmarks[a].x * w)
                ay = int(landmarks[a].y * h)
                bx = int(landmarks[b].x * w)
                by = int(landmarks[b].y * h)
                cv2.line(frame, (ax, ay), (bx, by), (0, 128, 255), 2)
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    # --- Draw hands + gesture logic ---
    left_wrist = None
    right_wrist = None

    if hand_result.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[i][0].category_name  # "Left" or "Right"

            # Draw connections
            for connection in HAND_CONNECTIONS:
                a, b = connection
                ax = int(hand_landmarks[a].x * w)
                ay = int(hand_landmarks[a].y * h)
                bx = int(hand_landmarks[b].x * w)
                by = int(hand_landmarks[b].y * h)
                cv2.line(frame, (ax, ay), (bx, by), (255, 0, 255), 2)
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

            wrist = hand_landmarks[WRIST]
            if handedness == "Left":
                left_wrist = wrist
            else:
                right_wrist = wrist

    # --- UDP packet (same logic as before) ---
    if left_wrist and right_wrist:
        lx, ly = left_wrist.x, left_wrist.y
        rx, ry = right_wrist.x, right_wrist.y

        mid_x = (lx + rx) / 2.0
        mid_y = (ly + ry) / 2.0
        dist = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)

        packet = f"{mid_x:.3f},{mid_y:.3f}".encode()
        sock.sendto(packet, (CHUMBO_IP, UDP_PORT))

        cv2.putText(
            frame,
            f"Mid=({mid_x:.2f},{mid_y:.2f}) Dist={dist:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pose_landmarker.close()
hand_landmarker.close()
