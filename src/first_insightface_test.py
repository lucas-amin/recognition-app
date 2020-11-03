"""1. Getting Started with Pre-trained Model from InsightFace
=======================================================


In this tutorial, we will demonstrate how to load a pre-trained model from :ref:`insightface-model-zoo`
and analyze faces from images.

Step by Step
------------------

Let's first try out a pre-trained insightface model with a few lines of python code.

First, please follow the `installation guide <../../index.html#installation>`__
to install ``MXNet`` and ``insightface`` if you haven't done so yet.
"""

import insightface
import urllib
import urllib.request
import cv2
import numpy as np

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

################################################################
#
# Then, we download and show the example image:

url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
img = url_to_image(url)

################################################################
# Init FaceAnalysis module by its default models
#

model = insightface.app.FaceAnalysis()

################################################################
# Use CPU to do all the job. Please change ctx-id to a positive number if you have GPUs
#

ctx_id = -1


################################################################
# Prepare the enviorment
# The nms threshold is set to 0.4 in this example.
#

model.prepare(ctx_id = ctx_id, nms=0.4)

################################################################
# Analysis faces in this image
#
faces = model.get(img)
for idx, face in enumerate(faces):
  print("Face [%d]:"%idx)
  print("\tage:%d"%(face.age))
  gender = 'Male'
  if face.gender==0:
    gender = 'Female'
  print("\tgender:%s"%(gender))
  print("\tembedding shape:%s"%face.embedding.shape)
  print("\tbbox:%s"%(face.bbox.astype(np.int).flatten()))
  print("\tlandmark:%s"%(face.landmark.astype(np.int).flatten()))
  print("")


