import os
import pytest
from src.Trainer.default_image_manager import TrainingImagesManager
from src.recognition_engine import RecognitionEngine
from test.tests_image_manager import TestsImageManager

@pytest.mark.order1
def test_recognize_person():
    engine = RecognitionEngine()

    image_manager = TrainingImagesManager()
    image_dict, names = image_manager.get_production_dataset_dict()

    for name in names:
        image_list = image_dict[name]

        for image in image_list:
            frame, result = engine.recognize_person(image)

            assert result[0]["name"] == name

@pytest.mark.order2
def test_include_new_person_on_dataset():
    engine = RecognitionEngine()

    test_image_manager = TestsImageManager()
    image_dict, names = test_image_manager.get_reduced_testing_dataset_dict()

    for name in names:
        image_list = image_dict[name]
        for image in image_list:
            path = engine.include_and_get_path_of_new_person_on_dataset(image, name, test=True)

            assert os.path.exists(path)

@pytest.mark.order3
def test_train():
    engine = RecognitionEngine()

    test_image_manager = TestsImageManager()
    image_dict, names = test_image_manager()

