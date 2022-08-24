import sys
import os
import glob
import platform
import traceback
import numpy as np
from sklearn import linear_model
from custom_types import RegressionMethod


def multi_file_extension_glob(base: str, extensions: str, recursive=False):
    """Performs glob matching with multiple filename extensions"""
    matches = []
    for extension in extensions:
        matches.extend(glob.glob(base + extension, recursive=recursive))
    return sorted(matches)


def get_extension_agnostic_path(base, extensions):
    """Get exact path to file given multiple possible filename extensions"""
    for extension in extensions:
        if os.path.exists(base + extension):
            return base + extension


def depth_from_file(path):
    """Read metric depth values from text files"""
    try:
        with open(path) as f:
            for line in f.read().strip().splitlines():
                return float(line)
    except Exception as e:
        raise RuntimeError(f"Failed to read depth from file '{path}': {e}")


def crop(img, crop_top, crop_bottom, crop_left, crop_right):
    """Crop images by configured amount from top, bottom, left and right"""
    ct = crop_top
    cb = -crop_bottom if crop_bottom > 0 else img.shape[0]
    cl = crop_left
    cr = -crop_right if crop_right > 0 else img.shape[1]
    return img[ct:cb, cl:cr]


def get_calibration_frame_dist(transect_dir, calibration_frame_id):
    """Try to get metric depth belonging to a calibration frames using multiple methods"""
    # first try wether the filename itself represents the distance in meters
    try:
        return int(calibration_frame_id)
    except:
        pass

    # otherwise, search for corresponding text files
    for depth_file_path in [
        os.path.join(
            transect_dir,
            "calibration_frames_cropped",
            f"{calibration_frame_id}_mean.txt",
        ),
        os.path.join(
            transect_dir,
            "calibration_frames_cropped",
            f"{calibration_frame_id}.txt",
        )
    ]:
        if os.path.exists(depth_file_path):
            return depth_from_file(depth_file_path)
    raise RuntimeError(f"Unable to find depth file for transect {transect_dir} and calibration frame id {calibration_frame_id}")


def calibrate(x, y, method, n=2, poly_deg=5):

    assert n in [1, 2]
    assert len(x) >= 2 and len(y) >= 2 and len(x) == len(y), f"inconsistent sample length in calibration: len(x)={len(x)}, len(y)={len(y)}"

    x_mask = x.mask if hasattr(x, "mask") else np.zeros_like(x, dtype=bool)
    y_mask = y.mask if hasattr(y, "mask") else np.zeros_like(y, dtype=bool)
    mask = x_mask | y_mask
    if np.mean(mask * 1) <= 0.5:
        x, y = (
            x[~mask],
            y[~mask],
        )

    x, y = x.reshape(-1), y.reshape(-1)

    if method == RegressionMethod.RANSAC:
        ransac = linear_model.RANSACRegressor()
        ransac.fit(np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1))
        c = ransac.predict(np.array([0]).reshape(-1, 1)) if n == 2 else 0
        m = ransac.predict(np.array([1]).reshape(-1, 1)) - c
        m, c = m.item(), c.item()
        return lambda data: m * data + c

    if method == RegressionMethod.LEASTSQUARES:
        try:
            if n == 2:
                A = np.vstack([x, np.ones(len(x))]).T
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]

                m, c = m.item(), c.item()
                return lambda data: m * data + c
            else:
                c = 0
                A = x.T
                m = np.linalg.lstsq(A, y, rcond=None)[0]

                m, c = m.item(), c.item()
                return lambda data: m * data + c
        except np.linalg.LinAlgError:
            return calibrate(x, y, method="RANSAC", n=n)

    if method == RegressionMethod.POLY:
        z, _ = np.polynomial.polynomial.polyfit(x, y, poly_deg, full=True)
        p = np.poly1d(z)
        return p

    if method == RegressionMethod.RANSAC_POLY:
        # adapted from https://gist.github.com/geohot/9743ad59598daf61155bf0d43a10838c

        n = 20  # minimum number of data points required to fit the model
        k = 100  # maximum number of iterations allowed in the algorithm
        t = 0.5  # threshold value to determine when a data point fits a model
        d = 100  # number of close data points required to assert that a model fits well to data
        f = 0.25  # fraction of close data points required

        besterr = np.inf
        bestfit = None
        for _ in range(k):
            maybeinliers = np.random.randint(len(x), size=n)
            maybemodel, _ = np.polynomial.polynomial.polyfit(
                x[maybeinliers], y[maybeinliers], poly_deg, full=True
            )
            maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], poly_deg)
            alsoinliers = np.abs(np.polyval(maybemodel, x) - y) < t
            if sum(alsoinliers) > d and sum(alsoinliers) > len(x) * f:
                bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], poly_deg)
                thiserr = np.sum(
                    np.abs(np.polyval(bettermodel, x[alsoinliers]) - y[alsoinliers])
                )
                if thiserr < besterr:
                    bestfit = bettermodel
                    besterr = thiserr

        if bestfit is None:
            raise ValueError("unable to calibrate using RANSAC_POLY")
        p = np.poly1d(bestfit)
        return p


def is_standalone():
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def get_onnxruntime_providers():
    if platform.system() != "Darwin":
        return [
            # "TensorrtExecutionProvider",  # TODO: reenable?
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        return [
            "CPUExecutionProvider",
        ]


def exception_to_str(e: Exception):
    return "".join(traceback.format_exception(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]))