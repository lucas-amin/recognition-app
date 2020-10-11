# Start tracking
import cv2
import dlib
import numpy as np

class Tracker():
    def __init__(self, facer):
        self.track_frames = 10
        self.trackers = []
        self.texts = []

        self.facer = facer
        self.recognizer = facer.facial_recogition

    def track(self, frame, frame_number):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = []

        if frame_number % self.track_frames == 0:
            bboxes = self.recognizer.detector.detect_faces(frame)
            self.trackers = []
            self.texts = []

            if len(bboxes) != 0:
                for bboxe in bboxes:
                    bbox = bboxe['box']
                    bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

                    landmarks = bboxe['keypoints']

                    nimg = self.recognizer.preprocess(frame, bbox, landmarks)

                    # Extract features from face
                    embedding = self.recognizer.embedding_model.get_feature(nimg).reshape(1, -1)

                    # Predict class
                    preds = self.recognizer.model.predict(embedding)
                    preds = preds.flatten()

                    name, probability = self.recognizer.check_prediction(preds, embedding)

                    results.append({"name": name, "probability": probability, "bbbox": bbox})

                    tracker = dlib.correlation_tracker()

                    rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                    tracker.start_track(rgb, rect)
                    self.trackers.append(tracker)
                    self.texts.append(name)

                    y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                    cv2.putText(frame, name, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        else:
            for tracker, text in zip(self.trackers, self.texts):
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        return results, frame