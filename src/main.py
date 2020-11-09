import sys

# Start streaming and recording
import time

import cv2

from src.Classifier.Facer import Facer
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_width = 600
save_height = int(600 / frame_width * frame_height)

facer = Facer(frame_width, frame_height, save_width, save_height)
frame_id = 0
while True:
    ret, frame = cap.read()
    frame_id += 1

    result, result_frame = facer.recognize_with_tracking(frame, frame_id=frame_id)

    cv2.imshow("Frame", result_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
