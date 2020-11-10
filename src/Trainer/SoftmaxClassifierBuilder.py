import os
import pickle

from keras.engine.saving import load_model
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import keras


class SoftmaxClassifierBuilder:
    DEFAULT_MODEL_PATH = "outputs/my_model.h5"
    DEFAULT_ENCODER_PATH = "outputs/le.pickle"
    TEST_MODEL_PATH = "outputs/my_test_model.h5"
    TEST_ENCODER_PATH = "outputs/le_test.pickle"

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self):
        self.model = Sequential()
        self.model.add(Dense(1024, activation='relu', input_shape=self.input_shape))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

        return self.model

    def save_default_model(self):
        # write the face recognition model to output
        self.model.save(self.DEFAULT_MODEL_PATH)
        f = open(self.DEFAULT_ENCODER_PATH, "wb")
        f.write(pickle.dumps(self.DEFAULT_ENCODER_PATH))
        f.close()

    def save_testing_model(self):
        # write the face recognition model to output
        self.model.save(self.TEST_MODEL_PATH)
        f = open(self.TEST_ENCODER_PATH, "wb")
        f.write(pickle.dumps(self.TEST_ENCODER_PATH))
        f.close()

    @staticmethod
    def get_classifier_from_file(softmax_model_path):
        corrected_model_path = os.path.join(os.path.dirname(__file__), softmax_model_path)
        classifier = load_model(corrected_model_path)
        return classifier
