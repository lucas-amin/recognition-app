import os

import insightface

from src.Classifier.FacialDetector import FacialDetector
from keras.models import load_model
import cv2
from keras import backend as K
import tensorflow as tf

from src.Classifier.SoftmaxResultChecker import SoftmaxResultChecker
from src.Classifier.Tracker import Tracker
from src.Classifier.face_embedding_extractor import FaceEmbeddingExtractor
from src.Trainer.SoftmaxClassifierBuilder import SoftmaxClassifierBuilder


class FacialRecognizer:
    # Initialize some useful arguments
    trackers = []
    texts = []
    frame_id = 0

    def __init__(self, facer):
        self.facer = facer
        self.tracker = Tracker()

        self.result_checker = SoftmaxResultChecker()

        # Initialize detector
        self.facial_detector = FacialDetector()

        # Initialize faces embedding model
        self.embedding_model = FaceEmbeddingExtractor()

        # Load the classifier model, determine if face is known
        self.softmax_model = SoftmaxClassifierBuilder.load_classifier_from_file("./outputs/my_model.h5")

        self.clean_tf_graph()

    def clean_tf_graph(self):
        # Clean up tensorflow graph
        self.__Session = K.get_session()
        self.__Graph = tf.get_default_graph()

    def reset(self):
        self.embedding_model.reset()
        self.tracker.reset()

    def recognize_threadsafe(self, frame):
        with self.__Session.as_default():
            with self.__Graph.as_default():
                frame, result = self.recognize(frame)

        return frame, result

    def recognize(self, frame):
        faces_bboxes, landmarks = self.facial_detector.get_faces_bbox_and_landmarks(frame)
        results = []

        if len(faces_bboxes) != 0:
            for bbox in faces_bboxes:
                output_frame = self.get_face_image(bbox, frame)

                name, probability = self.get_face_name(output_frame)

                result = self.get_result_dict(bbox, frame, name)

                self.draw_roi(frame, name, bbox)
                results.append(result)

        return frame, results

    def recognize_with_tracking(self, frame):
        results = []

        if self.frame_id % self.tracker.track_frames == 0:
            faces_bboxes, landmarks = self.facial_detector.get_faces_bbox_and_landmarks(frame)

            if len(faces_bboxes) != 0:
                for bbox in faces_bboxes:
                    output_frame = self.get_face_image(bbox, frame)

                    name, probability = self.get_face_name(output_frame)

                    result = self.get_result_dict(bbox, frame, name)

                    self.tracker.track_result_frame(name, frame, bbox)

                    self.draw_roi(frame, name, bbox)
                    results.append(result)
        else:
            self.tracker.track_non_result_frame(frame)

        return frame, results

    def get_face_name(self, output_frame):
        embedding = self.embedding_model.get_embedding(output_frame)
        preds = self.softmax_model.predict(embedding)
        name, probability = self.result_checker.check_prediction(preds, embedding)
        return name, probability

    def get_face_image(self, bbox, frame):
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        output_frame = frame[y:y + h, x:x + w]
        output_frame = cv2.resize(output_frame, (112, 112))

        return output_frame

    def get_result_dict(self, name, probability, bbox):
        face_result = {"name": name, "probability": probability, "bbbox": bbox}

        return face_result

    def draw_roi(self, frame, name, bbox):
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        y = int(y - 10 if y - 10 > 10 else y + 10)
        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)

