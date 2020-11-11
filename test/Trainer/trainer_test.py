from src.Classifier.SoftmaxResultChecker import SoftmaxResultChecker
from src.Trainer.RecognitionTrainer import RecognitionTrainer
from src.Trainer.SoftmaxClassifierBuilder import SoftmaxClassifierBuilder


def test_recognition_trainer():
    trainer = RecognitionTrainer()

    trainer.load_dataset()
    trainer.train()
    trainer.save_test_results()

    features_reader = trainer.get_features_reader()

    faces_dict = features_reader.get_dataset_faces()
    result_checker = SoftmaxResultChecker()

    classifier = SoftmaxClassifierBuilder.get_classifier_from_file(SoftmaxClassifierBuilder.TEST_MODEL_PATH)

    for face_dict in faces_dict:
        face = face_dict["image"]

        embedding = features_reader.get_face_embedding(face)

        prediction = classifier.predict(embedding)
        name, highest_probability = result_checker.check_prediction(prediction, embedding)

        print(name, highest_probability)

    assert False