import sys

# Start streaming and recording
import time

import cv2

from src.Facer.Facer import Facer
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    cap = camera.video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_width = 600
    save_height = int(600 / frame_width * frame_height)
    facer = Facer(frame_width, frame_height, save_width, save_height)
    frame_id = 0

    while True:
        camera_on, frame = cap.read()

        result, result_frame = facer.recognize_with_tracking(frame, frame_id=frame_id)

        frame_converted = camera.convert_opencv_to_base64(result_frame)

        frame_id += 1

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_converted + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
