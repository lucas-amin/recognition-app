from src.Classifier.face_embedding_extractor import FaceEmbeddingExtractor
from src.Classifier.softmax_classifier_manager import SoftmaxClassifierManager
from test.tests_image_manager import TestsImageManager

def get_image_dict():
    image_manager = TestsImageManager()
    image_dict = image_manager.get_testing_dataset_dict()
    names = image_dict.keys()
    return image_dict, names


def test_face_embedding_extractor():
    CPU_PROCESS_FRAMES_NUMBER = 30
    embedding_extractor = FaceEmbeddingExtractor(use_gpu=False)
    image_dict, names = get_image_dict()

    classifier_manager = SoftmaxClassifierManager()
    classifier_manager.train_test_classifier()




def test_gpu_face_embedding_extractor():
    embedding_extractor = FaceEmbeddingExtractor(use_gpu=True)

    image_manager = TestsImageManager()
    image_dict = image_manager.get_testing_dataset_dict()
    names = image_dict.keys()

    for name in names:
        image_list = image_dict[name]

        for image in image_list:
            embedding = embedding_extractor.get_embedding(image)
            assert len(embedding) == 512

