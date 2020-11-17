import sys
from pandas import np

from src.Trainer.default_image_manager import DatasetImagesManager
sys.path.insert(1, '../src/')
from camera import VideoCamera

def test_convert_numpymatrix_to_base64():
    camera = VideoCamera()
    imagemanager = DatasetImagesManager()

    image_dict, names = imagemanager.get_testing_dataset_dict()

    for name in names:
        list_image = image_dict[name]

        for image in list_image:
            base64_frame = camera.convert_numpymatrix_to_base64_string(image)
            decoded = camera.convert_base64bytes_to_numpymatrix(base64_frame)

            assert np.array_equal(decoded, image)
