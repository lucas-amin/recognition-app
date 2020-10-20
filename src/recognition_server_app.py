import base64
import io
import sys
from io import StringIO

import cv2
import imutils as imutils
from Facer.Facer import Facer
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from PIL import Image
from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index2.html')


@socketio.on('image')
def image(data_image):
    global frame_id
    frame = get_frame(data_image)

    facer = Facer.getFacerObject()

    # Process the image frame
    # frame = imutils.resize(frame, width=700)
    # frame = cv2.flip(frame, 1)

    result, frame = facer.recognize_with_tracking(frame, frame_id=0)

    imgencode = cv2.imencode('.jpg', frame)[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)


def get_frame(data_image):
    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    bytes_image = io.BytesIO(base64.b64decode(data_image))
    pillow_image = Image.open(bytes_image)

    ## converting RGB to BGR, as opencv standards
    frame = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)

    frame = cv2.resize(frame, (640, 480))

    return frame


if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', debug=True)
