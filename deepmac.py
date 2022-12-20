import sys
import os
import logging
import numpy as np
import cv2
import tensorflow as tf
from utils import is_standalone


class DeepMac:
    def __init__(self):
        weights_name = "deepmac_1024x1024_coco17.tflite"
        if is_standalone():
            model_path = os.path.join(sys._MEIPASS, "weights", weights_name)
        else:
            model_path = os.path.join("weights", weights_name)
        interpreter = tf.lite.Interpreter(model_path=model_path)
        self.model = interpreter.get_signature_runner()
        self.common_size = None


    def __call__(self, img, boxes):

        # skip model invocation if we have no given boxes
        if len(boxes) == 0:
            return np.array([])

        # prepare boxes
        boxes_input = boxes / np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        boxes_input = np.clip(boxes_input, 0, 1)
        boxes_input = np.array([[box[1], box[0], box[3], box[2]] for box in boxes_input], dtype=np.float32)

        # BGR to RGB
        img = img[..., ::-1]

        # resize
        if self.common_size is not None:
            img_input = cv2.resize(img, self.common_size, cv2.INTER_AREA)
        else:
            img_input = img

        # add batch dimension
        img_input = img_input[None, ...]
        boxes_input = boxes_input[None, ...]

        # compute
        output = self.model(input_tensor=img_input, boxes=boxes_input)

        masks = output["detection_masks"][0]
        return masks