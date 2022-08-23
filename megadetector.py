import sys
import os
import logging
import numpy as np
import cv2
import onnxruntime
from utils import get_onnxruntime_providers, is_standalone


class MegaDetectorLabel:
    ANIMAL = 0
    PERSON = 1
    VEHICLE = 2


class MegaDetector:
    def __init__(self):
        weights_name = "md_v5a.0.0.onnx"
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

        self.common_size = None


    def __call__(self, img):
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