import os
import pickle

import cv2
from imutils import paths
from src.Classifier.FacePreprocesser import FacePreprocesser
from src.Classifier.FacialDetector import FacialDetector
from src.Classifier.FacialRecognizer import FacialRecognizer
from src.Classifier.face_embedding_extractor import FaceEmbeddingExtractor
from src.file_utils import get_absolute_path


class DatasetFeaturesReader:
    def __init__(self):
        # Grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        training_directory = get_absolute_path("../datasets/train")
        self.imagePaths = list(paths.list_images(training_directory))
        self.embedding_path = get_absolute_path("Trainer/outputs/")
        self.embedding_filename = "embeddings.pickle"

        # Initialize the faces embedding model
        self.embedding_extractor = FaceEmbeddingExtractor()

        # Initialize our lists of extracted facial embeddings and corresponding people names
        self.known_embeddings = []
        self.known_names = []

        # Initialize the total number of faces processed
        self.registered_faces = 0
        self.facial_detector = FacialDetector()

    def get_dataset_faces(self):
        images = []
        for (i, imagePath) in enumerate(self.imagePaths):
            # extract the person name from the image path
            name = imagePath.split(os.path.sep)[-2]

            image = cv2.imread(imagePath)

            images.append({"label": name, "image": image})

        return images

    def extract_features_from_dataset(self):
        for (i, imagePath) in enumerate(self.imagePaths):
            # extract the person name from the image path
            name = imagePath.split(os.path.sep)[-2]

            image = cv2.imread(imagePath)

            face_embedding = self.get_face_embedding(image)

            self.register_result(face_embedding, name)

        self.save_to_picke_file()

    def extract_features_from_list(self, images, labels):
        # Loop over the imagePaths
        for i in range(len(images)):
            # extract the person name from the image path
            face_embedding = self.get_face_embedding(images[i])

            self.register_result(face_embedding, labels[i])

        self.save_to_picke_testfile()

    def save_to_picke_file(self):
        data = {"embeddings": self.known_embeddings, "names": self.known_names}
        f = open(self.get_pickle_filepath(), "wb")
        f.write(pickle.dumps(data))
        f.close()

    def save_to_picke_testfile(self):
        data = {"embeddings": self.known_embeddings, "names": self.known_names}
        f = open(self.get_test_filename(), "wb")
        f.write(pickle.dumps(data))
        f.close()

    def get_pickle_filepath(self):
        return self.embedding_path + self.embedding_filename

    def get_test_filename(self):
        return self.embedding_path + "testing_" + self.embedding_filename

    def delete_pickle_testfile(self):
        testfile_path = self.get_test_filename()

        if os.path.exists(testfile_path):
            os.remove(testfile_path)

    def register_result(self, face_embedding, name):
        self.known_names.append(name)
        self.known_embeddings.append(face_embedding[0])
        self.registered_faces += 1

    def get_face_embedding(self, image):
        image = self.facial_detector.get_single_cropped_face(image)
        face_embedding = self.embedding_extractor.get_embedding(image)
        return face_embedding

