import cv2
import insightface


class FaceEmbeddingExtractor:
    def __init__(self):
        self.setup_embedding_model()

    def setup_embedding_model(self, use_gpu=True):
        self.embedding_model = insightface.model_zoo.get_model('arcface_r100_v1')

        if use_gpu:
            self.embedding_model.prepare(ctx_id=0)
        else:
            self.embedding_model.prepare(ctx_id=-1)

        return self.embedding_model

    def get_embedding(self, input_frame):
        image = FaceEmbeddingExtractor.resize_to_input_size(input_frame)
        embedding = self.embedding_model.get_embedding(image)

        return embedding[0]

    @staticmethod
    def resize_to_input_size(image):
        image = cv2.resize(image, (112, 112))
        return image
