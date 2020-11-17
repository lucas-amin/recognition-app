import os
import cv2
from src.Classifier.Facer import Facer
from src.Classifier.softmax_classifier_manager import SoftmaxClassifierManager
from src.file_utils import get_absolute_path


class RecognitionEngine:
    PRODUCTION_DATASET_DIR = get_absolute_path("datasets/production_dataset/")
    STAGING_DATASET_DIR = get_absolute_path("datasets/staging_dataset/")
    reset_files = False

    def __init__(self, model):
        self.recognizer = Facer()
        self.softmax_manager = SoftmaxClassifierManager()
        self.model = model

    def recognize_person(self, frame):
        frame, result = self.recognizer.recognize_without_tracking(frame)
        return frame, result

    def include_new_person_on_dataset(self, image, name):
        if self.model == "STAGING":
            directory = RecognitionEngine.STAGING_DATASET_DIR + name + "/"
        elif self.model == "PRODUCTION":
            directory = RecognitionEngine.PRODUCTION_DATASET_DIR + name + "/"

        if not os.path.exists(directory):
            os.mkdir(directory)

        filename = str(len(os.listdir(directory))) + ".jpg"
        file_path = directory + filename

        cv2.imwrite(directory + filename, image)

        self.reset_files = True

        return file_path

    def load_model(self):
        if self.model == "STAGING":
            self.recognizer.set_softmax_model = self.softmax_manager.load_staging_classifier()
        if self.model == "DEFAULT":
            self.recognizer.set_softmax_model = self.softmax_manager.load_production_classifier()

    def reset_model(self):
        if self.model == "STAGING":
            self.recognizer.set_softmax_model = self.softmax_manager.load_staging_classifier()
        if self.model == "DEFAULT":
            self.recognizer.set_softmax_model = self.softmax_manager.load_production_classifier()

    def train_staging_model(self):
        self.softmax_manager.train_staging_classifier()
