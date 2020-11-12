from src.Classifier.SoftmaxResultChecker import SoftmaxResultChecker
from src.Trainer.SoftmaxClassifierBuilder import SoftmaxClassifierBuilder


class SoftmaxClassifierManager:
    def __init__(self):
        self.builder = SoftmaxClassifierBuilder()
        self.result_checker = SoftmaxResultChecker()

    def load_default_classifier(self):
        self.softmax_model = self.builder.load_default_classifier()

    def predict(self, embedding):
        preds = self.softmax_model.predict(embedding)
        name, probability = self.result_checker.check_prediction(preds, embedding)
        return name
