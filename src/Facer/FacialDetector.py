import insightface

class FacialDetector():
    def __init__(self):
        # Initialize detector
        self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
        self.detector.prepare(ctx_id=-1, nms=0.4)
        self.DETECTION_THRESHOLD = 0.5

        # bboxes, landmark = retinaface_detector.detect(frame, threshold=0.5, scale=1.0)
        # detect_tock = time.time()
        #
        # if len(bboxes) != 0:
        #     reco_tick = time.time()
        #     for bboxe in bboxes:
        #         x1, y1, w, h = int(bboxe[0]), int(bboxe[1]), int(bboxe[2]), int(bboxe[3])
        #         img = frame[y1:y1 + h, x1:x1 + w]
        #
        #         img = cv2.resize(img, (112, 112))

    def get_faces_bboxes(self, frame):
        faces_bboxes, landmarks = self.detector.detect(frame, threshold=self.DETECTION_THRESHOLD, scale=1.0)
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