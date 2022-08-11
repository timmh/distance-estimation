
import sys
import os
import json
import logging
import numpy as np
import cv2
import onnxruntime
from utils import get_onnxruntime_providers, is_standalone


class DPT:
    def __init__(self):
        weights_name = "dpt_hybrid-midas-6c3ec701.onnx"
        if is_standalone():
            with open(os.path.join(sys._MEIPASS, "weights", weights_name), "rb") as f:
                weight_bytes = f.read()
        else:
            with open(os.path.join("weights", weights_name), "rb") as f:
                weight_bytes = f.read()

        providers = get_onnxruntime_providers()
        try:
            self.session = onnxruntime.InferenceSession(
                weight_bytes,
                providers=providers,
            )
        except Exception as e:
            providers_str = ",".join(providers)
            logging.warn(f"Failed to create onnxruntime inference session with providers '{providers_str}', trying 'CPUExecutionProvider'")
            self.session = onnxruntime.InferenceSession(
                weight_bytes,
                providers=["CPUExecutionProvider"],
            )

        metadata = self.session.get_modelmeta().custom_metadata_map
        self.net_w, self.net_h = json.loads(metadata["ImageSize"])
        normalization = json.loads(metadata["Normalization"])
        self.prediction_factor = float(metadata["PredictionFactor"])
        self.mean = np.array(normalization["mean"])
        self.std = np.array(normalization["std"])
    
    def __call__(self, img):
        # BGR to RGB
        img = img[..., ::-1]

        # convert into 0..1 range
        img = img / 255.

        # resize
        img_input = cv2.resize(img, (self.net_h, self.net_w), cv2.INTER_AREA)

        # normalize
        img_input = (img_input - self.mean) / self.std

        # transpose from HWC to CHW
        img_input = img_input.transpose(2, 0, 1)

        # add batch dimension
        img_input = img_input[None, ...]

        # compute
        prediction = self.session.run(["output"], {"input": img_input.astype(np.float32)})[0][0]
        prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC)
        prediction *= self.prediction_factor

        return prediction