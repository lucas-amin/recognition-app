import pickle
import numpy as np
from src.file_utils import get_absolute_path


class SoftmaxResultChecker:
    cosine_threshold = 0.8
    proba_threshold = 0.85
    comparing_num = 5

    def __init__(self):
        # Load embeddings and labels
        embedding_path = get_absolute_path("src/Trainer/outputs/embeddings.pickle")
        self.data = pickle.loads(open(embedding_path, "rb").read())
        le_path = get_absolute_path("src/Trainer/outputs/le.pickle")
        self.le = pickle.loads(open(le_path, "rb").read())

        self.embeddings = np.array(self.data['embeddings'])
        self.labels = self.le.fit_transform(self.data['names'])

    def check_prediction(self, preds, embedding):
        # Get the highest accuracy embedded vector
        preds = preds.flatten()
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
        cos_similarity = SoftmaxResultChecker.CosineSimilarity(embedding, compare_embeddings)

        name = "unknown"

        # Set name as the highest probable person if it passes threshold
        if cos_similarity < SoftmaxResultChecker.cosine_threshold and \
                highest_probability > SoftmaxResultChecker.proba_threshold:
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
            cos_dist += SoftmaxResultChecker.findCosineDistance(test_vec, source_vec)
        return cos_dist / len(source_vecs)
