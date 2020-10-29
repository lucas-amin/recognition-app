import sys

sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import face_preprocess
import numpy as np
import face_model
import pickle
import time
import cv2
from keras import backend as K
import tensorflow as tf
#from tensorflow.python.keras import backend as K

class FacialRecognizer():
    # Initialize some useful arguments
    cosine_threshold = 0.8
    proba_threshold = 0.85
    comparing_num = 5
    trackers = []
    texts = []
    frames = 0

    def __init__(self, facer):
        self.facer = facer
        self.args = facer.args

        # Load embeddings and labels
        self.data = pickle.loads(open(self.args.embeddings, "rb").read())
        self.le = pickle.loads(open(self.args.le, "rb").read())

        self.embeddings = np.array(self.data['embeddings'])
        self.labels = self.le.fit_transform(self.data['names'])

        # Initialize detector
        self.detector = MTCNN()

        # Initialize faces embedding model
        self.embedding_model = face_model.FaceModel(self.args)

        # Load the classifier model, determine if face is known
        self.model = load_model(self.args.mymodel)

        # Clean up tensorflow graph
        self.__Session = K.get_session()
        self.__Graph = tf.get_default_graph()
        # self.__Graph.finalize()

    def reset(self):
        self.embedding_model = face_model.FaceModel(self.args)

    def recognize_threadsafe(self, frame):
        with self.__Session.as_default():
            with self.__Graph.as_default():
                frame, result = self.recognize(frame)

        return frame, result

    def recognize(self, frame):
        faces_bboxes = self.detector.detect_faces(frame)
        result = []

        if len(faces_bboxes) != 0:
            for bboxe in faces_bboxes:
                bbox = bboxe['box']
                bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

                preprocessed_image = self.preprocess(frame, bboxe['box'], bboxe['keypoints'])

                # Extract features from face
                embedding = self.embedding_model.get_feature(preprocessed_image).reshape(1, -1)

                # Predict class from recognition model
                preds = self.model.predict(embedding)
                preds = preds.flatten()

                name, probability = self.check_prediction(preds, embedding)

                result.append({"name": name, "probability": probability, "bbbox": bboxe['box']})

                y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                cv2.putText(frame, name, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        return frame, result

    @staticmethod
    def preprocess(frame, bbox, landmarks):
        bbox_out = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

        landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                              landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                              landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                              landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
        landmarks = landmarks.reshape((2, 5)).T

        transposed_image = face_preprocess.preprocess(frame, bbox_out, landmarks, image_size='112,112')

        transposed_image = cv2.cvtColor(transposed_image, cv2.COLOR_BGR2RGB)
        transposed_image = np.transpose(transposed_image, (2, 0, 1))

        return transposed_image

    def check_prediction(self, preds, embedding):
        # Get the highest accuracy embedded vector
        predicted_index = np.argmax(preds)
        highest_probability = preds[predicted_index]

        # This is a double-check, after the classifier said that the embedding belongs to the person X,
        # Actually gets some faces from this person and compares with the analyzed
        # Compare this vector to source class vectors to verify it is actual belong to this class
        match_class_idx = (self.labels == predicted_index)
        match_class_idx = np.where(match_class_idx)[0]
        selected_idx = np.random.choice(match_class_idx, self.comparing_num)
        compare_embeddings = self.embeddings[selected_idx]

        # Calculate cosine similarity
        cos_similarity = FacialRecognizer.CosineSimilarity(embedding, compare_embeddings)

        name = "unknown"

        # Set name as the highest probable person if it passes threshold
        if cos_similarity < FacialRecognizer.cosine_threshold and highest_probability > FacialRecognizer.proba_threshold:
            name = self.le.classes_[predicted_index]

        return name, highest_probability

    # Define distance function
    @staticmethod
    def findCosineDistance(vector1, vector2):
        """
        Calculate cosine distance between two vector
        """
        vec1 = vector1.flatten()
        vec2 = vector2.flatten()

        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    @staticmethod
    def CosineSimilarity(test_vec, source_vecs):
        """
        Verify the similarity of one vector to group vectors of one class
        """
        cos_dist = 0
        for source_vec in source_vecs:
            cos_dist += FacialRecognizer.findCosineDistance(test_vec, source_vec)
        return cos_dist / len(source_vecs)
