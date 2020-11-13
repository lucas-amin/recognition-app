import pickle
from keras.engine.saving import load_model

from src.file_utils import get_absolute_path


class SoftmaxFileManager:
    DEFAULT_MODEL_PATH = get_absolute_path("src/Trainer/outputs/my_model.h5")
    DEFAULT_LABEL_ENCODER_PATH = get_absolute_path("src/Trainer/outputs/le.pickle")
    TEST_MODEL_PATH = get_absolute_path("test/Trainer/outputs/my_test_model.h5")
    TEST_LABEL_ENCODER_PATH = get_absolute_path("test/Trainer/outputs/le_test.pickle")

    @staticmethod
    def save_default_model(model):
        SoftmaxFileManager.save_model(model,
                                      SoftmaxFileManager.DEFAULT_MODEL_PATH,
                                      SoftmaxFileManager.DEFAULT_LABEL_ENCODER_PATH)

    @staticmethod
    def save_testing_model(model):
        SoftmaxFileManager.save_model(model,
                                      SoftmaxFileManager.TEST_MODEL_PATH,
                                      SoftmaxFileManager.TEST_LABEL_ENCODER_PATH)

    @staticmethod
    def save_model(model, model_path, label_encoder_path):
        model.save(model_path)
        f = open(model_path, "wb")
        f.write(pickle.dumps(label_encoder_path))
        f.close()

    @staticmethod
    def load_default_classifier():
        classifier = load_model(SoftmaxFileManager.DEFAULT_MODEL_PATH)
        return classifier

    @staticmethod
    def load_test_classifier():
        classifier = load_model(SoftmaxFileManager.TEST_MODEL_PATH)
        return classifier

    @staticmethod
    def load_classifier_from_file(softmax_model_path):
        corrected_model_path = get_absolute_path(softmax_model_path)
        classifier = load_model(corrected_model_path)
        return classifier
