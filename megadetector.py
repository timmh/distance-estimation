import sys
import os
import logging
import numpy as np
import cv2
import onnxruntime
from utils import get_onnxruntime_providers, DownloadableWeights


class MegaDetectorLabel:
    ANIMAL = 0
    PERSON = 1
    VEHICLE = 2


class MegaDetector(DownloadableWeights):
    def __init__(self):
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True

        weights_url = "https://github.com/timmh/MegaDetectorLite/releases/download/v0.2/md_v5a.0.0.onnx"
        weights_md5 = "c2c93e4ed7e297eb650562df74341a25"
        weights_path = self.get_weights(weights_url, weights_md5)

        providers = get_onnxruntime_providers(enable_coreml=False)
        try:
            self.session = onnxruntime.InferenceSession(
                weights_path,
                providers=providers,
            )
        except Exception as e:
            providers_str = ",".join(providers)
            logging.warn(f"Failed to create onnxruntime inference session with providers '{providers_str}', trying 'CPUExecutionProvider'")
            self.session = onnxruntime.InferenceSession(
                weights_path,
                providers=["CPUExecutionProvider"],
            )

        self.common_size = None


    def __call__(self, img):
        # ensure model is loaded
        self._load_model()

        # BGR to RGB
        img = img[..., ::-1]

        # convert into 0..1 range
        img = img / 255.

        # resize
        if self.common_size is not None:
            img_input = cv2.resize(img, self.common_size, cv2.INTER_AREA)
        else:
            img_input = img

        # transpose from HWC to CHW
        img_input = img_input.transpose(2, 0, 1)

        # add batch dimension
        img_input = img_input[None, ...]

        # compute
        scores, labels, boxes = self.session.run(
            ["scores", "labels", "boxes"],
            {self.session.get_inputs()[0].name: img_input.astype(np.float32)
        })

        if self.common_size is not None:
            for box in boxes:
                box[0] = box[0] * img.shape[1] / self.common_size[0]
                box[1] = box[1] * img.shape[0] / self.common_size[1]
                box[2] = box[2] * img.shape[1] / self.common_size[0]
                box[3] = box[3] * img.shape[0] / self.common_size[1]

        return scores, labels, boxes