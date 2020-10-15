import base64
import cv2
from pandas import np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor = 0.6


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def release(self):
        self.video.release()

    @staticmethod
    def convert_numpymatrix_to_base64_string(frame):
        ret, jpeg = cv2.imencode('.bmp', frame)
        return jpeg

    @staticmethod
    def convert_base64bytes_to_numpymatrix(base64_ndarray):
        base64_string = base64_ndarray.tostring()
        img = cv2.imdecode(np.asarray(bytearray(base64_string), dtype="uint8"), cv2.IMREAD_COLOR)
        return img

    def get_frame(self):
        success, image = self.video.read()
        image = cv2.resize(image, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break
        ret, jpeg = cv2.imencode('.jpg', image)

        return jpeg.tobytes()
