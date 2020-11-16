import os

import cv2

from src.Classifier.Facer import Facer
from src.file_utils import get_absolute_path


class RecognitionEngine:
    PRODUCTION_DATASET_DIR = get_absolute_path("datasets/production_dataset/")
    PRODUCTION_TEST_DATASET_DIR = get_absolute_path("datasets/production_test_dataset/")

    def __init__(self):
        self.recognizer = Facer()

    def recognize_person(self, frame):
        frame, result = self.recognizer.recognize_without_tracking(frame)
        return frame, result

    def include_and_get_path_of_new_person_on_dataset(self, image, name, test):
        if test:
            directory = RecognitionEngine.PRODUCTION_TEST_DATASET_DIR + name + "/"
        else:
            directory = RecognitionEngine.PRODUCTION_DATASET_DIR + name + "/"

        if not os.path.exists(directory):
            os.mkdir(directory)

        filename = str(len(os.listdir(directory))) + ".jpg"
        file_path = directory + filename

        cv2.imwrite(directory + filename, image)

        return file_path

    def train(self):
        pass