import os
import pytest
from src.Trainer.default_image_manager import DatasetImagesManager
from src.recognition_engine import RecognitionEngine


@pytest.mark.order1
def test_recognize_person():
    engine = RecognitionEngine("PRODUCTION")

    image_manager = DatasetImagesManager()
    image_dict, names = image_manager.get_production_dataset_dict()

    for name in names:
        image_list = image_dict[name]

        for image in image_list:
            frame, result = engine.recognize_person(image)

            assert result[0]["name"] == name


@pytest.mark.order2
def test_include_new_person_on_dataset():
    engine = RecognitionEngine("STAGING")

    test_image_manager = DatasetImagesManager()
    test_image_manager.remove_testing_images_from_staging()

    image_dict, names = test_image_manager.get_reduced_testing_dataset_dict()

    for name in names:
        image_list = image_dict[name]
        for image in image_list:
            path = engine.include_new_person_on_dataset(image, name, model_name="STAGING")

            assert os.path.exists(path)

    test_image_manager.remove_testing_images_from_staging()


@pytest.mark.order3
def test_train_and_predict():
    image_manager = DatasetImagesManager()
    image_dict, names = image_manager.get_staging_dataset_dict()

    engine = RecognitionEngine()
    engine.train_staging_model()
    engine.load_model()

    for name in names:
        image_list = image_dict[name]
        for image in image_list:
            frame, result = engine.recognize_person(image)

            assert result[0]["name"] == name


@pytest.mark.order4
def test_load_and_predict():
    image_manager = DatasetImagesManager()
    image_dict, names = image_manager.get_staging_dataset_dict()

    engine = RecognitionEngine("STAGING")
    engine.load_model()

    for name in names:
        image_list = image_dict[name]
        for image in image_list:
            frame, result = engine.recognize_person(image)

            assert result[0]["name"] == name


@pytest.mark.order5
def test_load_predict_and_train():
    image_manager = DatasetImagesManager()
    image_manager.clean_staging_dataset()
    image_dict, names = image_manager.get_production_dataset_dict()

    engine = RecognitionEngine("STAGING")

    for name in names:
        image_list = image_dict[name]
        for image in image_list:
            engine.include_new_person_on_dataset(image, name)

            engine.train_staging_model()

            frame, result = engine.recognize_person(image)

            assert result[0]["name"] == name
