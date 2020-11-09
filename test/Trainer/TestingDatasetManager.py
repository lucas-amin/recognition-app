import cv2
from sklearn.datasets import fetch_lfw_people
import numpy as np

class TestingDatasetManager:
    def __init__(self):
        olivetti_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4, color=True)
        self.images = np.array(olivetti_dataset.images, dtype=np.int8)
        self.labels = olivetti_dataset.target

        # Print info on shapes and reshape where necessary
        print("Original x shape:", self.images.shape)
        print("New x shape:", self.labels.shape)
        print("y shape", self.labels.shape)

