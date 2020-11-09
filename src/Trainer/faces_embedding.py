import sys    def get

import insightface
from src.Classifier.FacialDetector import FacialDetector
from imutils import paths
import argparse
import pickle
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument("--dataset", default="../../datasets/train",
                help="Path to training dataset")
ap.add_argument("--embeddings", default="./outputs/embeddings.pickle")

# Argument of insightface
ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()

# Grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args.dataset))

# Initialize the faces embedder
embedding_model = insightface.model_zoo.get_model('arcface_r100_v1')
embedding_model.prepare(ctx_id=-1)

# Initialize our lists of extracted facial embeddings and corresponding people names
knownEmbeddings = []
knownNames = []

# Initialize the total number of faces processed
total = 0
facial_detector = FacialDetector()


def resize_to_input_size(image):
    height, width, channels = image.shape
    if height != 112:
        image = cv2.resize(image, (112, height))
    if width != 112:
        image = cv2.resize(image, (112, 112))
    return image


# Loop over the imagePaths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the image
    image = cv2.imread(imagePath)

    # Get only the cropped face image
    image = facial_detector.get_single_cropped_face(image)

    # convert face to RGB color
    image = resize_to_input_size(image)

    # Get the face embedding vector
    face_embedding = embedding_model.get_embedding(image)

    # add the name of the person + corresponding face
    # embedding to their respective list
    knownNames.append(name)
    knownEmbeddings.append(face_embedding[0])
    total += 1

print(total, " faces embedded")

# save to output
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args.embeddings, "wb")
f.write(pickle.dumps(data))
f.close()
