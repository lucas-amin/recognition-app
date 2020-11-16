import insightface


class FacialDetector():
    def __init__(self):
        # Initialize detector
        self.detector = FacialDetector.get_detector_model()
        self.DETECTION_THRESHOLD = 0.5

    @staticmethod
    def get_detector_model(use_gpu=False):
        detector = insightface.model_zoo.get_model('retinaface_r50_v1')

        if use_gpu:
            detector.prepare(ctx_id=0, nms=0.4)
        else:
            detector.prepare(ctx_id=-1, nms=0.4)

        return detector

    def get_faces_bbox_and_landmarks(self, frame):
        detection_result, landmarks = self.detector.detect(frame, threshold=self.DETECTION_THRESHOLD, scale=1.0)

        faces_bboxes = []
        for face_bbox in detection_result:
            int_bbox = int(face_bbox[0]), int(face_bbox[1]), int(face_bbox[2]), int(face_bbox[3])
            faces_bboxes.append(int_bbox)

        return faces_bboxes, landmarks

    def get_single_cropped_face(self, frame):
        bbox_dict, landmarks = self.detector.detect(frame, threshold=0.4, scale=1.0)

        if len(bbox_dict) == 1:
            bbox = bbox_dict[0]
            x1, y1, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            output_frame = frame[y1:y1 + h, x1:x1 + w]

        else:
            print("This frame does not have one face")
            output_frame = frame

        return output_frame
