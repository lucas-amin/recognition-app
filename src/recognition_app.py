import sys

from Facer.Facer import Facer
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    cap = camera.video
    frame_id = 0
    facer = Facer.getFacerObject()

    try:
        while cap.isOpened():
            camera_on, frame = cap.read()

            result, result_frame = facer.recognize_with_tracking(frame, frame_id=frame_id)

            frame_converted = camera.convert_numpymatrix_to_base64_string(result_frame)

            frame_id += 1

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_converted.tobytes() + b'\r\n\r\n')

    finally:
        camera.video.release()


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
