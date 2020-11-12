from src.Classifier.softmax_classifier_manager import SoftmaxClassifierManager


def test_softmax_classifier_manager():
    classifier_manager = SoftmaxClassifierManager()

    classifier_manager.load_default_classifier()

    embedding_generator.get_embedding(frame)

    name_result = classifier_manager.predict(frame)

    assert name_result == expected_name_result
