from src.Classifier.face_embedding_extractor import FaceEmbeddingExtractor
from test.tests_image_manager import TestsImageManager


def test_face_embedding_extractor():
    embedding_extractor = FaceEmbeddingExtractor()
    image_manager = TestsImageManager()
    image_dict = image_manager.get_testing_dataset_images()
    names = image_dict.keys()

    for name in names:
        image_list = image_dict[name]

        for image in image_list:
            embedding = embedding_extractor.get_embedding(image)
            assert len(embedding) == 512

def get_embedding_test():
    pass
