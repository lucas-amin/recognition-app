from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from src.Trainer.EmbeddingManager import EmbeddingManager
from src.Trainer.softmax_classifier_loader import SoftmaxFileManager

class SoftmaxModelTrainer:
    # Initialize Softmax training model arguments
    BATCH_SIZE = 64
    EPOCHS = 10

    def __init__(self):
        self.features_reader = None
        self.dataset = None

    def train_and_get_model(self, dataset):
        embeddings, labels = self.get_and_load_embeddings(dataset=dataset)
        model = self.setup_and_train(embeddings, labels)
        return model

    def get_and_load_embeddings(self, dataset):
        self.features_reader = EmbeddingManager()

        if dataset is "TEST":
            self.embeddings, self.labels = self.features_reader.load_test_embeddings()
            self.dataset = "TEST"
        elif dataset is "PRODUCTION":
            self.embeddings, self.labels = self.features_reader.load_default_embeddings()
            self.dataset = "PRODUCTION"
        elif dataset is "STAGING":
            self.embeddings, self.labels = self.features_reader.load_staging_embeddings()
            self.dataset = "STAGING"

        return self.embeddings, self.labels

    def setup_and_train(self, embeddings, labels):
        self.setup_classifier(embeddings, labels)

        # Create KFold
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

        # Train
        for train_idx, valid_idx in cv.split(self.embeddings):
            X_train, X_val = self.embeddings[train_idx], self.embeddings[valid_idx]
            y_train, y_val = self.labels[train_idx], self.labels[valid_idx]

            his = self.model.fit(X_train, y_train,
                                 batch_size=SoftmaxModelTrainer.BATCH_SIZE,
                                 epochs=SoftmaxModelTrainer.EPOCHS, verbose=1,
                                 validation_data=(X_val, y_val))

            history['acc'] += his.history['acc']
            history['val_acc'] += his.history['val_acc']
            history['loss'] += his.history['loss']
            history['val_loss'] += his.history['val_loss']

        return self.model

    def encode_labels_and_embeddings(self, embeddings, labels):
        # Encode the labels
        self.label_encoder = LabelEncoder()
        raw_labels = self.label_encoder.fit_transform(labels)
        raw_labels = raw_labels.reshape(-1, 1)
        one_hot_encoder = OneHotEncoder()

        self.num_classes = len(np.unique(raw_labels))
        self.labels = one_hot_encoder.fit_transform(raw_labels).toarray()
        self.embeddings = np.array(embeddings)

    def setup_classifier(self, embeddings, labels):
        self.encode_labels_and_embeddings(embeddings, labels)

        self.input_shape = self.embeddings.shape[1]

        # Build softmax classifier
        self.build_classifier()

    def build_classifier(self):
        self.model = Sequential()
        self.model.add(Dense(1024, activation='relu', input_shape=(self.input_shape,)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    def get_model(self):
        return self.model

    def save_results(self):
        if self.dataset == "TEST":
            SoftmaxFileManager.save_testing_model(self.model, self.label_encoder)
        elif self.dataset == "PRODUCTION":
            SoftmaxFileManager.save_default_model(self.model, self.label_encoder)
        elif self.dataset == "STAGING":
            SoftmaxFileManager.save_staging_model(self.model, self.label_encoder)
        elif self.dataset is None:
            print("Logging: dataset not defined")
