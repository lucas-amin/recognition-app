from src.Classifier.face_embedding_extractor import FaceEmbeddingExtractor
from test.tests_image_manager import TestsImageManager


def test_face_embedding_extractor():
    CPU_PROCESS_FRAMES_NUMBER = 30
    embedding_extractor = FaceEmbeddingExtractor(use_gpu=False)
    image_manager = TestsImageManager()
    image_dict, names = image_manager.get_testing_dataset_dict()

    for name in names:
        image_list = image_dict[name][:CPU_PROCESS_FRAMES_NUMBER]

        for image in image_list:
            embedding = embedding_extractor.get_face_embedding(image)
            assert len(embedding) == 1
            assert len(embedding[0]) == 512

def test_gpu_face_embedding_extractor():
    embedding_extractor = FaceEmbeddingExtractor(use_gpu=True)

    image_manager = TestsImageManager()
    image_dict, names = image_manager.get_testing_dataset_dict()

    for name in names:
        image_list = image_dict[name]

        for image in image_list:
            embedding = embedding_extractor.get_face_embedding(image)
            assert len(embedding) == 1
            assert len(embedding[0]) == 512
