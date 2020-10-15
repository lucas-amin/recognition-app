from .FacialRecognizer import FacialRecognizer
import argparse
import cv2

from .Tracker import Tracker


class Facer:
    facer_object = None

    def __init__(self):
        self.args = Facer.parse_arguments()
        self.facial_recognizer = FacialRecognizer(self)
        self.tracker = Tracker(self)
        self.frame_id = 0

    @staticmethod
    def getFacerObject():
        if Facer.facer_object is None:
            Facer.facer_object = Facer()

        Facer.facer_object.reset()

        return Facer.facer_object

    def reset(self):
        self.tracker.reset()
        self.facial_recognizer.reset()

    def recognize_with_tracking(self, frame, frame_id):
        result, result_frame = self.tracker.track(frame, frame_id)

        return result, result_frame

    def recognize_without_tracking(self, frame, frame_id):
        result, result_frame = self.tracker.track(frame, frame_id)

        return result, result_frame

    @staticmethod
    def parse_arguments():
        ap = argparse.ArgumentParser()

        ap.add_argument("--mymodel", default="outputs/my_model.h5",
                        help="Path to recognizer model")
        ap.add_argument("--le", default="outputs/le.pickle",
                        help="Path to label encoder")
        ap.add_argument("--embeddings", default="outputs/embeddings.pickle",
                        help='Path to embeddings')
        ap.add_argument("--video-out", default="../datasets/videos_output/stream_test.mp4",
                        help='Path to output video')
        ap.add_argument('--image-size', default='112,112', help='')
        ap.add_argument('--model', default='../insightface/models/model-y1-test2/model,0', help='path to load model.')
        ap.add_argument('--ga-model', default='', help='path to load model.')
        ap.add_argument('--gpu', default=0, type=int, help='gpu id')
        ap.add_argument('--det', default=0, type=int,
                        help='mtcnn option, 1 means using R+O, 0 means detect from begining')
        ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

        args = ap.parse_args()

        return args