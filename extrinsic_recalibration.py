import logging
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime

from utils import DownloadableWeights, get_onnxruntime_providers


@dataclass
class HomographyEstimate:
    homography: np.ndarray
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    reprojection_error: float
    reused_previous: bool = False


class LightGlueONNX(DownloadableWeights):
    def __init__(self, max_image_size=1024):
        self.max_image_size = max_image_size
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True

        weights_url = "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.0/superpoint_1024_lightglue_end2end.onnx"
        weights_path = self.get_weights(weights_url)

        providers = get_onnxruntime_providers(enable_coreml=False)
        try:
            self.session = onnxruntime.InferenceSession(
                weights_path,
                providers=providers,
            )
        except Exception:
            providers_str = ",".join(providers)
            logging.warn(f"Failed to create onnxruntime inference session with providers '{providers_str}', trying 'CPUExecutionProvider'")
            self.session = onnxruntime.InferenceSession(
                weights_path,
                providers=["CPUExecutionProvider"],
            )

    def __call__(self, img0, img1):
        self._load_model()

        input_tensors, scales0, scales1 = self._prepare_inputs(img0, img1)
        output_names = [output.name for output in self.session.get_outputs()]
        outputs = self.session.run(output_names, input_tensors)
        output_map = {name: output for name, output in zip(output_names, outputs)}
        pts0, pts1, scores = self._parse_outputs(output_map)

        pts0[:, 0] /= scales0[0]
        pts0[:, 1] /= scales0[1]
        pts1[:, 0] /= scales1[0]
        pts1[:, 1] /= scales1[1]
        return pts0, pts1, scores

    def _prepare_inputs(self, img0, img1):
        inputs = self.session.get_inputs()
        tensor0, scales0 = self._preprocess(img0, inputs[0].shape)
        tensor1, scales1 = self._preprocess(img1, inputs[-1].shape)

        if len(inputs) == 1:
            return {inputs[0].name: np.concatenate([tensor0, tensor1], axis=0)}, scales0, scales1
        if len(inputs) == 2:
            return {inputs[0].name: tensor0, inputs[1].name: tensor1}, scales0, scales1

        raise RuntimeError(f"Unsupported LightGlue ONNX input count: {len(inputs)}")

    def _preprocess(self, img, input_shape):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray = gray.astype(np.float32) / 255.0

        target_h, target_w = self._target_size(gray.shape, input_shape)
        if gray.shape != (target_h, target_w):
            gray = cv2.resize(gray, (target_w, target_h), cv2.INTER_AREA)

        tensor = gray[None, None, :, :].astype(np.float32)
        return tensor, (target_w / img.shape[1], target_h / img.shape[0])

    def _target_size(self, shape, input_shape):
        h, w = shape
        fixed_h, fixed_w = self._fixed_hw(input_shape)
        if fixed_h is not None and fixed_w is not None:
            return fixed_h, fixed_w

        scale = min(1.0, self.max_image_size / max(h, w))
        target_h = max(8, int(round(h * scale / 8) * 8))
        target_w = max(8, int(round(w * scale / 8) * 8))
        return target_h, target_w

    def _fixed_hw(self, input_shape):
        if input_shape is None or len(input_shape) < 4:
            return None, None
        h, w = input_shape[-2], input_shape[-1]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return h, w
        return None, None

    def _parse_outputs(self, output_map):
        outputs = {name.lower(): value for name, value in output_map.items()}
        keypoints0 = self._get_output(outputs, ["keypoints0", "kpts0", "keypoints_0", "kpts_0"])
        keypoints1 = self._get_output(outputs, ["keypoints1", "kpts1", "keypoints_1", "kpts_1"])
        matched0 = self._get_output(outputs, ["matched_keypoints0", "mkpts0", "matches_keypoints0"], required=False)
        matched1 = self._get_output(outputs, ["matched_keypoints1", "mkpts1", "matches_keypoints1"], required=False)

        if matched0 is not None and matched1 is not None:
            scores = self._get_output(outputs, ["scores", "mscores", "matching_scores", "mscores0"], required=False)
            matched0 = self._points(matched0)
            matched1 = self._points(matched1)
            return matched0, matched1, self._scores(scores, len(matched0))

        matches = self._get_output(outputs, ["matches", "matches0", "matches_0"])
        scores = self._get_output(outputs, ["scores", "mscores", "matching_scores", "mscores0"], required=False)
        keypoints0 = self._points(keypoints0)
        keypoints1 = self._points(keypoints1)

        matches = np.asarray(matches)
        if matches.ndim == 3:
            matches = matches[0]
        if matches.ndim == 2 and matches.shape[0] == 1:
            matches = matches[0]
        if matches.ndim == 1:
            valid = matches >= 0
            idx0 = np.nonzero(valid)[0]
            idx1 = matches[valid].astype(np.int64)
            score_values = self._scores(scores, len(matches))[valid]
        else:
            valid = np.all(matches >= 0, axis=1)
            idx0 = matches[valid, 0].astype(np.int64)
            idx1 = matches[valid, 1].astype(np.int64)
            score_values = self._scores(scores, len(matches))[valid]

        valid_idx = (idx0 < len(keypoints0)) & (idx1 < len(keypoints1))
        return keypoints0[idx0[valid_idx]], keypoints1[idx1[valid_idx]], score_values[valid_idx]

    def _get_output(self, outputs, names, required=True):
        for name in names:
            if name in outputs:
                return outputs[name]
        if required:
            raise RuntimeError(f"LightGlue ONNX output not found. Expected one of {names}, got {list(outputs.keys())}")
        return None

    def _points(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 3:
            points = points[0]
        return points.reshape(-1, 2)

    def _scores(self, scores, length):
        if scores is None:
            return np.ones(length, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(scores) == length:
            return scores
        return np.ones(length, dtype=np.float32)


class SIFTMatcher:
    def __init__(self, max_features=2048):
        self.sift = cv2.SIFT_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def __call__(self, img0, img1):
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) if img0.ndim == 3 else img0
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
        keypoints0, descriptors0 = self.sift.detectAndCompute(gray0, None)
        keypoints1, descriptors1 = self.sift.detectAndCompute(gray1, None)
        if descriptors0 is None or descriptors1 is None or len(descriptors0) < 2 or len(descriptors1) < 2:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

        pairs = self.matcher.knnMatch(descriptors0, descriptors1, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        pts0 = np.float32([keypoints0[m.queryIdx].pt for m in good])
        pts1 = np.float32([keypoints1[m.trainIdx].pt for m in good])
        scores = np.float32([1.0 - min(1.0, m.distance / 512.0) for m in good])
        return pts0, pts1, scores


class ExtrinsicRecalibrator:
    def __init__(self):
        self.lightglue = LightGlueONNX()
        self.fallback_matcher = SIFTMatcher()
        self.use_fallback = False
        self.previous_homography = None

    def reset(self):
        self.previous_homography = None

    def estimate(self, baseline_img, img):
        try:
            if self.use_fallback:
                pts0, pts1, scores = self.fallback_matcher(baseline_img, img)
            else:
                pts0, pts1, scores = self.lightglue(baseline_img, img)
        except Exception as e:
            logging.warn(f"LightGlue-ONNX matching failed, falling back to OpenCV SIFT for extrinsic recalibration: {e}")
            self.use_fallback = True
            pts0, pts1, scores = self.fallback_matcher(baseline_img, img)

        estimate = self._estimate_homography(pts0, pts1, scores, img.shape[0:2])
        if estimate is not None and self._is_quality_acceptable(estimate, img.shape[0:2]):
            self.previous_homography = estimate.homography
            return estimate

        if self.previous_homography is not None:
            return HomographyEstimate(
                homography=self.previous_homography,
                num_matches=0 if estimate is None else estimate.num_matches,
                num_inliers=0 if estimate is None else estimate.num_inliers,
                inlier_ratio=0.0 if estimate is None else estimate.inlier_ratio,
                reprojection_error=np.inf if estimate is None else estimate.reprojection_error,
                reused_previous=True,
            )
        return None

    def _estimate_homography(self, pts0, pts1, scores, image_shape):
        if len(pts0) < 8 or len(pts1) < 8:
            return None

        order = np.argsort(scores)[::-1] if len(scores) == len(pts0) else np.arange(len(pts0))
        pts0 = pts0[order]
        pts1 = pts1[order]
        homography, inlier_mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, 4.0)
        if homography is None or inlier_mask is None:
            return None

        inliers = inlier_mask.ravel().astype(bool)
        if not np.any(inliers):
            return None

        projected = cv2.perspectiveTransform(pts1[inliers, None, :], homography)[:, 0, :]
        errors = np.linalg.norm(projected - pts0[inliers], axis=1)
        return HomographyEstimate(
            homography=homography,
            num_matches=len(pts0),
            num_inliers=int(np.sum(inliers)),
            inlier_ratio=float(np.mean(inliers)),
            reprojection_error=float(np.median(errors)),
        )

    def _is_quality_acceptable(self, estimate, image_shape):
        h, w = image_shape
        if estimate.num_inliers < 16 or estimate.inlier_ratio < 0.25 or estimate.reprojection_error > 6.0:
            return False
        return is_homography_sane(estimate.homography, (h, w))


def is_homography_sane(homography, image_shape):
    h, w = image_shape
    if not np.all(np.isfinite(homography)):
        return False

    corners = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ])
    warped = cv2.perspectiveTransform(corners[None, :, :], homography)[0]
    if not np.all(np.isfinite(warped)):
        return False

    source_area = cv2.contourArea(corners)
    warped_area = abs(cv2.contourArea(warped.astype(np.float32)))
    area_ratio = warped_area / max(source_area, 1.0)
    if area_ratio < 0.2 or area_ratio > 5.0:
        return False

    max_displacement = np.max(np.linalg.norm(warped - corners, axis=1))
    return max_displacement <= max(h, w) * 0.75


def warp_image(img, homography, target_shape):
    return cv2.warpPerspective(
        img,
        homography,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def warp_depth(depth, homography, target_shape):
    return cv2.warpPerspective(
        depth,
        homography,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

