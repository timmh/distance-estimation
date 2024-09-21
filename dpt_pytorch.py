import numpy as np
from utils import DownloadableWeights, condition_disparity


class DPTPyTorch(DownloadableWeights):
    def __init__(self):
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True

        import torch
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.model = self.model.eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def __call__(self, img, optimize=True):
        import torch

        # ensure model is loaded
        self._load_model()

        # BGR to RGB
        img = img[..., ::-1]

        # transform
        img_input = self.transform(img)

        # compute
        with torch.inference_mode():
            if self.device == torch.device("cuda"):
                self.model = self.model.to(memory_format=torch.channels_last)
                self.model = self.model.half()
                img_input = img_input.to(self.device)
                img_input = img_input.to(memory_format=torch.channels_last)
                img_input = img_input.half()

            prediction = self.model(img_input)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        prediction = prediction.cpu().numpy().astype(np.float32)
        prediction = condition_disparity(prediction)
        return prediction