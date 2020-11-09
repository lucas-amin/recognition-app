import sys

from pandas import np

sys.path.insert(1, '../src/')

import cv2
import pytest
from camera import VideoCamera


images = ["arya-bran.jpeg", "daenerys-jon_snow.jpeg", "daenerys-jon_snow-cersei.jpg", "unknown.jpg"]
@pytest.mark.parametrize("img_name", images)
def test_convert_numpymatrix_to_base64(img_name):
    camera = VideoCamera()
    frame = cv2.imread("../data/" + img_name)

    base64_frame = camera.convert_numpymatrix_to_base64_string(frame)

    decoded = camera.convert_base64bytes_to_numpymatrix(base64_frame)

    assert np.array_equal(decoded, frame)
