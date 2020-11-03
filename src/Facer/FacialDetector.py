import insightface

class FacialDetector():
    def __init__(self):
        # Initialize detector
        self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
        self.detector.prepare(ctx_id=0, nms=0.4)
        self.DETECTION_THRESHOLD = 0.5

    def get_faces_bboxes(self, frame):
        faces_bboxes, landmarks = self.detector.detect(frame, threshold=self.DETECTION_THRESHOLD, scale=1.0)
        faces_bboxes[0][0] = int(faces_bboxes[0][0])
        faces_bboxes[0][1] = int(faces_bboxes[0][1])
        faces_bboxes[0][2] = int(faces_bboxes[0][2])
        faces_bboxes[0][3] = int(faces_bboxes[0][3])

        return faces_bboxes

    def get_single_cropped_face(self, frame):
        bbox_dict, landmarks = self.detector.detect(frame, threshold=0.5, scale=1.0)

        if len(bbox_dict) == 1:
            bbox = bbox_dict[0]
            x1, y1, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            output_frame = frame[y1:y1+h, x1:x1+w]

        else:
            print("This frame has more than 1 face")
            output_frame = None

        return output_frame