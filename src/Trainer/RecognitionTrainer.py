from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from src.Trainer.DatasetFeaturesReader import DatasetFeaturesReader
from src.Trainer.SoftmaxClassifierBuilder import SoftmaxClassifierBuilder


class RecognitionTrainer:
    # Initialize Softmax training model arguments
    BATCH_SIZE = 64
    EPOCHS = 25

    def __init__(self):
        self.features_reader = None

    def load_dataset(self):
        self.load_labels()
        self.setup_classifier()

    def load_labels(self):
        self.features_reader = self.get_features_reader()

        self.encode_labels()

    def get_features_reader(self):
        if self.features_reader is None:
            self.features_reader = DatasetFeaturesReader()
            self.features_reader.extract_features_from_dataset()
        return self.features_reader

    def encode_labels(self):
        # Encode the labels
        le = LabelEncoder()
        raw_labels = le.fit_transform(self.features_reader.known_names)
        raw_labels = raw_labels.reshape(-1, 1)
        one_hot_encoder = OneHotEncoder()

        self.num_classes = len(np.unique(raw_labels))
        self.labels = one_hot_encoder.fit_transform(raw_labels).toarray()
        self.embeddings = np.array(self.features_reader.known_embeddings)

    def setup_classifier(self):
        input_shape = self.embeddings.shape[1]

        # Build sofmax classifier
        self.softmax_classifier = SoftmaxClassifierBuilder(input_shape=(input_shape,), num_classes=self.num_classes)
        self.softmax_classifier.build()

    def train(self):
        # Create KFold
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

        # Train
        for train_idx, valid_idx in cv.split(self.embeddings):
            X_train, X_val = self.embeddings[train_idx], self.embeddings[valid_idx]
            y_train, y_val = self.labels[train_idx], self.labels[valid_idx]

            his = self.softmax_classifier.model.fit(X_train, y_train,
                                                    batch_size=RecognitionTrainer.BATCH_SIZE,
                                                    epochs=RecognitionTrainer.EPOCHS, verbose=1,
                                                    validation_data=(X_val, y_val))

            print(his.history['acc'])
            history['acc'] += his.history['acc']
            history['val_acc'] += his.history['val_acc']
            history['loss'] += his.history['loss']
            history['val_loss'] += his.history['val_loss']

    def get_model(self):
        return self.softmax_classifier.model

    def save_default_results(self):
        self.softmax_classifier.save_default_model()

    def save_test_results(self):
        self.softmax_classifier.save_testing_model()