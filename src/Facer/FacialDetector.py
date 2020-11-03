from mtcnn.mtcnn import MTCNN

class FacialDetector():
    def __init__(self):
        # Initialize detector
        self.detector = MTCNN()

    def get_faces_bboxes(self, frame):
        faces_bboxes = self.detector.detect_faces(frame)

        return faces_bboxes
