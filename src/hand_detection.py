import cv2
import mediapipe as mp
from pathlib import Path
from recognizer import extract_features

model_path = Path("models/hand_landmarker.task")
print(model_path.exists())
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print(cap.isOpened())

# success, frame = cap.read()
# print(success)

# cv2.imshow("Sign Bridge", frame)
# cv2.waitKey(0)

timestamp_ms = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms += 33
    results = detector.detect_for_video(mp_image, timestamp_ms)

    if results.hand_landmarks:
        hand = results.hand_landmarks[0]
        features = extract_features(hand)
        print(len(features))

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Sign Bridge", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()