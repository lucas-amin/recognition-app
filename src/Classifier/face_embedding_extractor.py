import cv2
import insightface
from src.Classifier.FacialDetector import FacialDetector


class FaceEmbeddingExtractor:
    def __init__(self, use_gpu=True):
        self.setup_embedding_model(use_gpu=use_gpu)
        self.facial_detector = FacialDetector()

    def setup_embedding_model(self, use_gpu=True):
        self.embedding_model = insightface.model_zoo.get_model('arcface_r100_v1')

        if use_gpu:
            self.embedding_model.prepare(ctx_id=0)
        else:
            self.embedding_model.prepare(ctx_id=-1)

        return self.embedding_model

    def get_face_embedding(self, image):
        image = FaceEmbeddingExtractor.resize_to_input_size(image)
        face_embedding = self.embedding_model.get_embedding(image)
        return face_embedding

    @staticmethod
    def resize_to_input_size(image):
        image = cv2.resize(image, (112, 112))
        return image
