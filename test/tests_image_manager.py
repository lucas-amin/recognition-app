import os
import cv2
from src.file_utils import get_absolute_path

class TestsImageManager:

    def __init__(self):
        self.path = get_absolute_path("./test/UnitTestsImages/")

    def get_testing_dataset_dict(self):
        names = os.listdir(self.path)
        image_dict = {}

        for name in names:
            image_names = os.listdir(self.path + name)
            self.get_folder_images(image_dict, image_names, name)

        return image_dict, names

    def get_folder_images(self, image_dict, image_names, name):
        image_dict[name] = []
        for image_name in image_names:
            image = cv2.imread(self.path + name + "/" + image_name)
            image_dict[name].append(image)
