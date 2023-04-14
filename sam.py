
import sys
import os
import logging
import numpy as np
import cv2
import onnxruntime
from utils import get_onnxruntime_providers, is_standalone


class SAM:
    def __init__(self):
        for session_name in ["encoder", "decoder"]:
            weights_name = f"sam_vit_b_01ec64_{session_name}.onnx"
            if is_standalone():
                with open(os.path.join(sys._MEIPASS, "weights", weights_name), "rb") as f:
                    weight_bytes = f.read()
            else:
                with open(os.path.join("weights", weights_name), "rb") as f:
                    weight_bytes = f.read()

            providers = get_onnxruntime_providers()
            try:
                session = onnxruntime.InferenceSession(
                    weight_bytes,
                    providers=providers,
                )
                setattr(self, f"{session_name}_session", session)
            except Exception as e:
                providers_str = ",".join(providers)
                logging.warn(f"Failed to create onnxruntime inference session with providers '{providers_str}', trying 'CPUExecutionProvider'")
                self.session = onnxruntime.InferenceSession(
                    weight_bytes,
                    providers=["CPUExecutionProvider"],
                )
        self.image_size = (1024, 1024)
        self.pixel_mean = np.array([123.675, 116.28, 103.53])
        self.pixel_std = np.array([58.395, 57.12, 57.375])

    def __call__(self, img, boxes):
        img = img[..., ::-1]
        original_size = img.shape[0:2]
        img = cv2.copyMakeBorder(img, 0, max(0, img.shape[1] - img.shape[0]), 0, max(0, img.shape[0] - img.shape[1]), cv2.BORDER_CONSTANT)
        fx, fy = self.image_size[1] / img.shape[1], self.image_size[0] / img.shape[0]
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)

        img = (img - self.pixel_mean[None, None, :]) / self.pixel_std[None, None, :]
        img = img[None, ...]
        img = img.transpose(0, 3, 1, 2)
        img = img.astype(np.float32)

        image_embedding = self.encoder_session.run(None, {"x": img})[0]

        mask_list = []
        for box in boxes:
            onnx_box_coords = box.reshape(-1, 2, 2)
            onnx_box_labels = np.array([2,3])
            onnx_coord = onnx_box_coords.astype(np.float32)
            onnx_label = onnx_box_labels[None, :].astype(np.float32)
            onnx_coord[..., 0] *= fy
            onnx_coord[..., 1] *= fx

            masks, _, _ = self.decoder_session.run(None, {
                "image_embeddings": image_embedding,
                "point_coords": onnx_coord.astype(np.float32),
                "point_labels": onnx_label.astype(np.float32),
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.array([0], dtype=np.float32),
                "orig_im_size": np.array(original_size, dtype=np.float32)
            })
            masks = masks > 0.0
            mask = masks[0, 0]
            mask_list += [mask]

        return np.array(mask_list)