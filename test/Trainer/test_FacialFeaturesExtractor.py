# 1.2 Our dataset is here
import sys

import cv2
from TestingDatasetManager import TestingDatasetManager

sys.path.insert(1, '../../src/')
from Trainer.DatasetFeaturesReader import DatasetFeaturesReader

import pytest

@pytest.fixture
def dataset():
    dataset_manager = TestingDatasetManager()

    return dataset_manager

def test_extract_features(dataset):
    features_reader = DatasetFeaturesReader()

    names = dataset.labels
    images = dataset.images

    for image in images:
        cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    features_reader.extract_features_from_list(images, names)
