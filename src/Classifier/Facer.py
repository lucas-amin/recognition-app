from .FacialRecognizer import FacialRecognizer

class Facer:
    facer_object = None

    def __init__(self):
        self.facial_recognizer = FacialRecognizer(self)
        self.frame_id = 0

    @staticmethod
    def get_facer_object():
        if Facer.facer_object is None:
            Facer.facer_object = Facer()

        return Facer.facer_object

    @staticmethod
    def get_and_reset_facer_object():
        if Facer.facer_object is None:
            Facer.facer_object = Facer()

        Facer.facer_object.reset()

        return Facer.facer_object

    def reset(self):
        self.facial_recognizer.reset()

    def recognize_with_tracking(self, frame):
        frame, result = self.facial_recognizer.recognize_with_tracking(frame)

        return frame, result

    def recognize_without_tracking(self, frame):
        frame, results = self.facial_recognizer.recognize(frame)

        return frame, results

    def recognize_without_tracking_threadsafe(self, frame):
        frame, result = self.facial_recognizer.recognize_threadsafe(frame)
        return frame, result
