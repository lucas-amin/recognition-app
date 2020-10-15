from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
# import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import face_image
import face_preprocess


def do_flip(data):
    for idx in xrange(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    # model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, args):
        ctx = self.construct_arguments(args)
        image_size = self.construct_image_size(args)
        self.construct_model(args, ctx, image_size)
        detector = self.construct_detector(args, ctx)
        self.detector = detector

    def construct_arguments(self, args):
        self.args = args
        ctx = mx.cpu(0)
        self.threshold = args.threshold
        self.det_minsize = 50
        self.det_threshold = [0.6, 0.7, 0.8]
        return ctx

    def construct_image_size(self, args):
        _vec = args.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.image_size = image_size
        return image_size

    def construct_model(self, args, ctx, image_size):
        self.model = None
        self.ga_model = None
        if len(args.model) > 0:
            self.model = get_model(ctx, image_size, args.model, 'fc1')
        if len(args.ga_model) > 0:
            self.ga_model = get_model(ctx, image_size, args.ga_model, 'fc1')

    def construct_detector(self, args, ctx):
        mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')

        if args.det == 0:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True,
                                     threshold=self.det_threshold)
        else:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True,
                                     threshold=[0.0, 0.0, 0.2])
        return detector

    def get_input(self, face_img):
        ret = self.detector.detect_face(face_img, det_type=self.args.det)
        if ret is None:
            return None
        bbox, points = ret
        if bbox.shape[0] == 0:
            return None
        bbox = bbox[0, 0:4]
        points = points[0, :].reshape((2, 5)).T

        nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        return aligned

    def get_feature(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)

        # TODO error when the facemodel is reutilized
        embedding_np_2 = self.extract_embedding_with_multiple_context()

        return embedding_np_2

    def extract_embedding_with_single_context(self):
        embedding = self.model.get_outputs(merge_multi_context=False)[0][0]
        embedding_np = embedding.asnumpy()
        embedding_np = sklearn.preprocessing.normalize(embedding_np).flatten()
        return embedding_np

    def extract_embedding_with_multiple_context(self):
        embedding = self.model.get_outputs()[0]
        embedding_np = embedding.asnumpy()
        embedding_np = sklearn.preprocessing.normalize(embedding_np).flatten()
        return embedding_np

    def get_ga(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.ga_model.forward(db, is_train=False)
        ret = self.ga_model.get_outputs()[0].asnumpy()
        g = ret[:, 0:2].flatten()
        gender = np.argmax(g)
        a = ret[:, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))

        return gender, age
