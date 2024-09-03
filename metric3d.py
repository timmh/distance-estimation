
import sys
import os
import json
import logging
import numpy as np
import cv2
import onnxruntime
from utils import get_onnxruntime_providers, DownloadableWeights


class Metric3D(DownloadableWeights):
    def __init__(self):
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True

        weights_url = "https://github.com/timmh/Metric3D/releases/download/v0.1/metric3d_vit_small.onnx"
        weights_md5 = "f620d1b8d70dd3cd8652b82cfe9f9a77"
        weights_path = self.get_weights(weights_url, weights_md5)

        providers = get_onnxruntime_providers()
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

        metadata = self.session.get_modelmeta().custom_metadata_map
        self.net_w, self.net_h = json.loads(metadata["ImageSize"])
        normalization = json.loads(metadata["Normalization"])
        self.prediction_factor = float(metadata["PredictionFactor"])
        self.mean = np.array(normalization["mean"])
        self.std = np.array(normalization["std"])
    
    def __call__(self, img):
        # ensure model is loaded
        self._load_model()

        # BGR to RGB
        img = img[..., ::-1]

        # resize
        img_input = cv2.resize(img, (self.net_w, self.net_h), cv2.INTER_LINEAR)

        # normalize
        img_input = (img_input - self.mean) / self.std

        # transpose from HWC to CHW
        img_input = img_input.transpose(2, 0, 1)

        # add batch dimension
        img_input = img_input[None, ...]

        # compute
        prediction = self.session.run(["pred_depth"], {"image": img_input.astype(np.float32)})[0][0][0]
        prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC)
        prediction *= self.prediction_factor

        # into disparity
        prediction = np.clip(prediction, 1e-6, np.inf) ** -1

        return prediction