import os
import cv2
from src.file_utils import get_absolute_path


class DatasetImagesManager:
    def __init__(self):
        self.production_path = get_absolute_path("./datasets/production_dataset/")
        self.staging_path = get_absolute_path("./datasets/staging_dataset/")
        self.testing_path = get_absolute_path("./test/UnitTestsImages/")

    def get_production_dataset_dict(self):
        names = os.listdir(self.production_path)
        image_dict = {}

        for name in names:
            image_names = os.listdir(self.production_path + name)
            image_dict = self.get_folder_images(image_dict, image_names, name, self.production_path)

        return image_dict, names

    def get_staging_dataset_dict(self):
        names = os.listdir(self.staging_path)
        image_dict = {}

        for name in names:
            image_names = os.listdir(self.production_path + name)
            image_dict = self.get_folder_images(image_dict, image_names, name, self.staging_path)

        return image_dict, names

    def get_testing_dataset_dict(self):
        names = os.listdir(self.testing_path)
        image_dict = {}

        for name in names:
            image_names = os.listdir(self.testing_path + name)
            image_dict = self.get_folder_images(image_dict, image_names, name, self.testing_path)

        return image_dict, names

    def get_reduced_testing_dataset_dict(self):
        names = os.listdir(self.testing_path)
        image_dict = {}

        for name in names:
            image_names = os.listdir(self.testing_path + name)[:3]
            image_dict = self.get_folder_images(image_dict, image_names, name, self.testing_path)

        return image_dict, names

    def clean_staging_dataset(self):
        names = os.listdir(self.staging_path)

        for name in names:
            image_names = os.listdir(self.staging_path + name)
            for image_name in image_names:
                os.remove(self.staging_path + name + "/" + image_name)
            os.rmdir(self.staging_path + name)

    def remove_testing_images_from_staging(self):
        production_list = os.listdir(self.production_path)
        staging_list = os.listdir(self.staging_path)

        for item in staging_list:
            if item not in production_list:
                folder = self.staging_path + item
                for item in os.listdir(folder):
                    os.remove(folder + "/" + item)
                os.rmdir(folder)

    def get_folder_images(self, image_dict, image_names, name, path):
        image_dict[name] = []
        for image_name in image_names:
            image = cv2.imread(path + name + "/" + image_name)
            image_dict[name].append(image)
        return image_dict
