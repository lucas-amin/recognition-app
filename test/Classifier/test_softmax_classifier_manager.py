import cv2

from src.Classifier.FacialDetector import FacialDetector
from src.Classifier.face_embedding_extractor import FaceEmbeddingExtractor
from src.Classifier.softmax_classifier_manager import SoftmaxClassifierManager
from src.Trainer.default_image_manager import TrainingImagesManager
from test.tests_image_manager import TestsImageManager


def iterate_manager_on_images(classifier_manager, embedding_extractor, image_dict, names):
    for name in names:
        image_list = image_dict[name]

        for image in image_list:
            detector = FacialDetector()

            cropped_image = detector.get_single_cropped_face(image)

            if cropped_image is None:
                cropped_image = image

            embedding = embedding_extractor.get_face_embedding(cropped_image)
            result_name, probability = classifier_manager.predict(embedding)

            assert name == result_name


def test_load_default_softmax_classifier():
    classifier_manager = SoftmaxClassifierManager()
    classifier_manager.load_default_classifier()
    embedding_extractor = FaceEmbeddingExtractor(use_gpu=True)

    image_manager = TrainingImagesManager()
    image_dict, names = image_manager.get_testing_dataset_dict()

    iterate_manager_on_images(classifier_manager, embedding_extractor, image_dict, names)


def test_train_default_softmax_classifier():
    # classifier_manager = SoftmaxClassifierManager()
    # classifier_manager.train_default_classifier()
    # embedding_extractor = FaceEmbeddingExtractor(use_gpu=True)
    # 
    # image_manager = TestsImageManager()
    # image_dict, names = image_manager.get_testing_dataset_dict()
    # 
    # iterate_manager_on_images(classifier_manager, embedding_extractor, image_dict, names)
    pass

def test_train_test_softmax_classifier():
    classifier_manager = SoftmaxClassifierManager()
    classifier_manager.train_test_classifier()
    embedding_extractor = FaceEmbeddingExtractor(use_gpu=True)

    image_manager = TestsImageManager()
    image_dict, names = image_manager.get_testing_dataset_dict()

    iterate_manager_on_images(classifier_manager, embedding_extractor, image_dict, names)


def test_loading_face_embedding_extractor():
    classifier_manager = SoftmaxClassifierManager()
    classifier_manager.load_test_classifier()
    embedding_extractor = FaceEmbeddingExtractor(use_gpu=True)

    image_manager = TestsImageManager()
    image_dict, names = image_manager.get_testing_dataset_dict()

    iterate_manager_on_images(classifier_manager, embedding_extractor, image_dict, names)



