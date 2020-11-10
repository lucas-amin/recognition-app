from src.Trainer.RecognitionTrainer import RecognitionTrainer
from src.Trainer.SoftmaxClassifier import SoftmaxClassifierBuilder


def test_recognition_trainer():
    trainer = RecognitionTrainer()

    trainer.load_dataset()
    trainer.train()
    trainer.save_test_results()

    classifier = SoftmaxClassifierBuilder.get_classifier_from_file(SoftmaxClassifierBuilder.TEST_MODEL_PATH)

    assert False
