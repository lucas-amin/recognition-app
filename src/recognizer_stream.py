from Classifier.Facer import Facer
import cv2

facer = Facer.getAndResetFacerObject()

cap = cv2.VideoCapture(1)

status, frame = cap.read()
frame_id = 0

while status is True:
    status, frame = cap.read()

    frame, result = facer.recognize_with_tracking(frame)

    cv2.imshow("result_frame", frame)
    cv2.waitKey(1)

    frame_id += 1
