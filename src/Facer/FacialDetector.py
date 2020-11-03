from mtcnn.mtcnn import MTCNN
import numpy as np

class FacialDetector():
    def __init__(self):
        # Initialize detector
        self.detector = MTCNN()

    def get_faces_bboxes(self, frame):
        faces_bboxes = self.detector.detect_faces(frame)
        return faces_bboxes

    def get_single_cropped_face(self, frame):
        bbox_dict = self.detector.detect_faces(frame)

        if len(bbox_dict) == 1:
            bbox = bbox_dict[0]['box']
            x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            output_frame = frame[y1:y1+h, x1:x1+w]

        else:
            print("This frame has more than 1 face")
            output_frame = None

        return output_frame