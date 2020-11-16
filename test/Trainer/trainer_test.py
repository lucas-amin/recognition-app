from src.Trainer.SoftmaxModelTrainer import SoftmaxModelTrainer
from src.Classifier.face_embedding_extractor import FaceEmbeddingExtractor
from test.tests_image_manager import TestsImageManager

CPU_PROCESS_FRAMES_NUMBER = 30
embedding_extractor = FaceEmbeddingExtractor(use_gpu=True)
image_manager = TestsImageManager()

def test_recognition_trainer():
    trainer = SoftmaxModelTrainer()
    model = trainer.train_and_get_model_with_test_dataset()
    trainer.save_results()

    image_dict, names = image_manager.get_testing_dataset_dict()

    for name in names:
        image_list = image_dict[name][:CPU_PROCESS_FRAMES_NUMBER]

        for image in image_list:
            embedding = embedding_extractor.get_face_embedding(image)
            result = model.predict(embedding)
            assert len(result) == len(name)

