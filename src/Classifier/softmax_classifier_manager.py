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

    def load_production_classifier(self):
        self.softmax_model = SoftmaxFileManager.load_default_classifier()

    def train_production_classifier(self):
        self.softmax_model = self.trainer.train_and_get_model("PRODUCTION")

    def load_unittest_classifier(self):
        self.softmax_model = SoftmaxFileManager.load_test_classifier()

    def train_unittest_classifier(self):
        self.softmax_model = self.trainer.train_and_get_model("TESTING")

    def load_staging_classifier(self):
        self.softmax_model = SoftmaxFileManager.load_staging_classifier()

    def train_staging_classifier(self):
        self.softmax_model = self.trainer.train_and_get_model("STAGING")
        self.trainer.save_results()

    def reset_classifier(self, model):
        SoftmaxFileManager.delete_model(model)
        self.softmax_model = self.trainer.train_and_get_model(model)
        self.trainer.save_results()
