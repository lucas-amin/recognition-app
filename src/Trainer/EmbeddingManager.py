import os
import pickle

import cv2
from imutils import paths
import numpy as np

from src.Classifier.FacialDetector import FacialDetector
from src.Classifier.face_embedding_extractor import FaceEmbeddingExtractor
from src.file_utils import get_absolute_path


class EmbeddingManager:
    def __init__(self):
        # Grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        self.set_default_directories()
        self.set_test_directories()
        self.set_staging_directories()

        # Initialize the faces embedding model
        self.embedding_extractor = FaceEmbeddingExtractor()

        # Initialize our lists of extracted facial embeddings and corresponding people names
        self.known_embeddings = []
        self.known_names = []

        # Initialize the total number of faces processed
        self.registered_faces = 0
        self.facial_detector = FacialDetector()

    def set_default_directories(self):
        training_directory = get_absolute_path("./datasets/train")
        self.default_image_paths = list(paths.list_images(training_directory))
        self.default_embedding_path = get_absolute_path("./src/Trainer/outputs/embeddings.pickle")

    def set_test_directories(self):
        training_directory = get_absolute_path("./test/UnitTestsImages/")
        self.test_images_paths = list(paths.list_images(training_directory))
        self.test_embedding_path = get_absolute_path("test/Trainer/outputs/embeddings.pickle")

    def set_staging_directories(self):
        training_directory = get_absolute_path("datasets/staging_dataset/")
        self.staging_images_paths = list(paths.list_images(training_directory))
        self.staging_embedding_path = get_absolute_path("datasets/staging_output/embeddings.pickle")

    def extract_embeddings_from_dataset(self, path):
        for (i, imagePath) in enumerate(path):
            # extract the person name from the image path
            name = imagePath.split(os.path.sep)[-2]

            image = cv2.imread(imagePath)

            face_embedding = self.get_face_embedding(image)

            self.register_result(face_embedding, name)

        return self.known_embeddings, self.known_names

    def load_test_embeddings(self):
        if self.testing_embedding_file_exists():
            self.known_embeddings, self.known_names = self.load_embeddings_from_files(self.test_embedding_path)
        else:
            self.known_embeddings, self.known_names = self.extract_embeddings_from_dataset(self.test_images_paths)
            self.save_to_picke_testfiles()

        return self.known_embeddings, self.known_names

    def load_staging_embeddings(self):
        if self.staging_embedding_file_exists():
            self.load_embeddings_from_files(self.staging_embedding_path)
        else:
            self.extract_embeddings_from_dataset(self.staging_images_paths)
            self.save_staging_files()

        return self.known_embeddings, self.known_names

    def load_default_embeddings(self):
        if self.default_embeddings_file_exists():
            self.load_embeddings_from_files(self.default_embedding_path)
        else:
            self.extract_embeddings_from_dataset(self.default_image_paths)
            self.save_to_picke_files()

        return self.known_embeddings, self.known_names

    def testing_embedding_file_exists(self):
        return os.path.exists(self.test_embedding_path)

    def staging_embedding_file_exists(self):
        return os.path.exists(self.staging_embedding_path)

    def default_embeddings_file_exists(self):
        return os.path.exists(self.default_embedding_path)

    def load_embeddings_from_files(self, test_embeddings_path):
        data = pickle.loads(open(test_embeddings_path, "rb").read())
        self.known_embeddings = np.array(data['embeddings'])
        self.known_names = np.array(data['names'])

        return self.known_embeddings, self.known_names

    def extract_features_from_list(self, images, labels):
        # Loop over the imagePaths
        for i in range(len(images)):
            # extract the person name from the image path
            face_embedding = self.get_face_embedding(images[i])

            self.register_result(face_embedding, labels[i])

        self.save_to_picke_testfiles()

    def save_to_picke_files(self):
        data = {"embeddings": self.known_embeddings, "names": self.known_names}
        f = open(self.default_embedding_path, "wb")
        f.write(pickle.dumps(data))
        f.close()

    def save_to_picke_testfiles(self):
        data = {"embeddings": self.known_embeddings, "names": self.known_names}
        f = open(self.test_embedding_path, "wb")
        f.write(pickle.dumps(data))
        f.close()

    def save_staging_files(self):
        data = {"embeddings": self.known_embeddings, "names": self.known_names}
        f = open(self.staging_embedding_path, "wb")
        f.write(pickle.dumps(data))
        f.close()

    def delete_pickle_testfile(self):
        if os.path.exists(self.test_embedding_path):
            os.remove(self.test_embedding_path)

    def register_result(self, face_embedding, name):
        self.known_names.append(name)
        self.known_embeddings.append(face_embedding[0])
        self.registered_faces += 1

    def get_face_embedding(self, image):
        image = self.facial_detector.get_single_cropped_face(image)
        face_embedding = self.embedding_extractor.get_face_embedding(image)
        return face_embedding
