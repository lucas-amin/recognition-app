from src.Trainer.SoftmaxModelTrainer import SoftmaxModelTrainer
from src.Classifier.SoftmaxResultChecker import SoftmaxResultChecker
from src.Trainer.softmax_classifier_loader import SoftmaxFileManager


class SoftmaxClassifierManager:
    def __init__(self):
        self.trainer = SoftmaxModelTrainer()
        self.result_checker = SoftmaxResultChecker()

    def predict(self, embedding):
        preds = self.softmax_model.predict(embedding)
        name, probability = self.result_checker.check_prediction(preds, embedding)
        return name, probability

    def load_default_classifier(self):
        self.softmax_model = SoftmaxFileManager.load_default_classifier()

    def train_default_classifier(self):
        self.softmax_model = self.trainer.train_and_get_model_with_default_dataset()

    def load_test_classifier(self):
        self.softmax_model = SoftmaxFileManager.load_test_classifier()

    def train_test_classifier(self):
        self.softmax_model = self.trainer.train_and_get_model_with_test_dataset()
