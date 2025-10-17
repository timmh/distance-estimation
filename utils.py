import sys
import os
import glob
import platform
import traceback
import argparse
import enum
import urllib.request
import re
from contextlib import contextmanager
import hashlib
import cv2
import numpy as np
from sklearn import linear_model
import scipy.ndimage as ndimage
from platformdirs import PlatformDirs
from custom_types import RegressionMethod


dirs = PlatformDirs("DistanceEstimation", "timmh")
random_seed = 42


@contextmanager
def random_seed_manager(seed=random_seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


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


def resize(img, target_size):
    """Resize img to target_size if img is not already at target_size"""
    if img.shape[0:2] != target_size[0:2]:
        img = cv2.resize(img, (target_size[1], target_size[0]), cv2.INTER_LINEAR)
    return img


def get_calibration_frame_dist(transect_dir, calibration_frame_id):
    """Try to get metric depth belonging to a calibration frames using multiple methods"""
    # first try wether the filename itself represents the distance in meters
    try:
        return float(re.match(r"^(\d+(?:\.\d*)?)(?:m|cm|dm|ft|in)?$", calibration_frame_id)[1])
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
    with random_seed_manager():

        assert n in [1, 2]
        assert len(x) >= 2 and len(y) >= 2 and len(x) == len(y), f"inconsistent sample length in calibration: len(x)={len(x)}, len(y)={len(y)}"

        x_mask = x.mask if hasattr(x, "mask") else np.zeros_like(x, dtype=bool)
        y_mask = y.mask if hasattr(y, "mask") else np.zeros_like(y, dtype=bool)
        mask = x_mask | y_mask
        x, y = (
            x[~mask],
            y[~mask],
        )

        x, y = x.reshape(-1), y.reshape(-1)

        if method == RegressionMethod.RANSAC:
            def is_model_valid(model, X_, y_):
                return (model.coef_ > 0).all()
            estimator = linear_model.LinearRegression(positive=True)
            ransac = linear_model.RANSACRegressor(estimator=estimator, is_model_valid=is_model_valid, random_state=random_seed)
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


def calibrate_v0(x, y, method, n=2, poly_deg=5):
    with random_seed_manager():

        assert n in [1, 2]

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
            try:
                ransac = linear_model.RANSACRegressor()
                ransac.fit(np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1))
                c = ransac.predict(np.array([0]).reshape(-1, 1)) if n == 2 else 0
                m = ransac.predict(np.array([1]).reshape(-1, 1)) - c
                m, c = m.item(), c.item()
                return lambda data: m * data + c
            except ValueError:
                print(f"Failed RANSAC calibration. Retrying with LEASTSQUARES.")
                A = np.vstack([x, np.ones(len(x))]).T
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]

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
                return do_calibrate(x, y, method="RANSAC", n=n)

        if method == RegressionMethod.POLY:
            z, _ = np.polynomial.polynomial.polyfit(x, y, poly_deg, full=True)
            p = np.poly1d(z)
            return p

        if method == RegressionMethod.RANSAC_POLY:
            # adapted from https://gist.github.com/geohot/9743ad59598daf61155bf0d43a10838c

            # n – minimum number of data points required to fit the model
            # k – maximum number of iterations allowed in the algorithm
            # t – threshold value to determine when a data point fits a model
            # d – number of close data points required to assert that a model fits well to data
            # f – fraction of close data points required
            n = 20
            k = 100
            t = 0.5
            d = 100
            f = 0.25

            besterr = np.inf
            bestfit = None
            for kk in range(k):
                maybeinliers = np.random.randint(len(x), size=n)
                # maybemodel, _ = np.polynomial.polynomial.polyfit(
                #     x[maybeinliers], y[maybeinliers], poly_deg, full=True
                # )
                import warnings

                warnings.simplefilter("ignore", np.RankWarning)

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


def get_onnxruntime_providers(enable_coreml=True):
    if platform.system() != "Darwin":
        return [
            # "TensorrtExecutionProvider",  # TODO: reenable?
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        return (["CoreMLExecutionProvider"] if enable_coreml else []) + ["CPUExecutionProvider"]


def exception_to_str(e: Exception):
    return "".join(traceback.format_exception(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]))


# taken from https://github.com/python/cpython/issues/69247#issuecomment-1308082792
class EnumActionLowerCase(argparse.Action):
    """
    Action to accept Enums by lowercase name and output an enum value.

    >>> class Item(enum.Enum):
    ...     Foo = 1
    ...     Bar = 2
    ...
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('item', type=Item, action=EnumActionLowerCase)
    >>> args = parser.parse_args("--item foo".split())
    >>> assert args.item == Item.Foo

    Source: https://stackoverflow.com/a/60750535/79125
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError(
                "type must be assigned an Enum when using EnumActionLowerCase"
            )
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumActionLowerCase")

        # Generate choices from the Enum
        lower_names = tuple(e.name.lower() for e in enum_type)
        unique_names = set(lower_names)
        if len(lower_names) > len(unique_names):
            raise ValueError(
                "enum names must be lowercase unique when using EnumActionLowerCase"
            )
        kwargs.setdefault("choices", lower_names)

        super(EnumActionLowerCase, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, value, option_string=None):
        # Find the matching enum member
        value = next(e for e in self._enum if e.name.lower() == value)
        setattr(namespace, self.dest, value)


def md5sum_from_filepath(filepath, chunksize=8192):
    with open(filepath, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(chunksize)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(chunksize)

    return file_hash.hexdigest()


class DownloadableWeights:
    def get_weights(self, weights_url, md5sum=None):
        download_dir = os.path.join(dirs.user_cache_dir, "weights")
        filename = weights_url.split("/")[-1]
        filepath = os.path.join(download_dir, filename)
        try:
            if os.path.exists(filepath):
                assert md5sum is None or md5sum_from_filepath(filepath) == md5sum
                return filepath
            else:
                os.makedirs(download_dir, exist_ok=True)
                urllib.request.urlretrieve(weights_url, filepath)
                assert md5sum is None or md5sum_from_filepath(filepath) == md5sum
                return filepath
        except Exception as e:
            os.unlink(filepath)
            raise RuntimeError(f"Failed retrieving weight '{filename}'. Please try again. Full exception: {e}")
            raise e


def blur_and_downsample(img, calibration_downsampling_factor=1/8, calibration_blur_sigma=41):
    mask = img.mask if (hasattr(img, "mask") and img.mask.shape != ()) else None
    img = img if mask is None else img.data

    img = ndimage.gaussian_filter(img, sigma=calibration_blur_sigma)
    img = cv2.resize(
        img,
        None,
        fx=calibration_downsampling_factor,
        fy=calibration_downsampling_factor,
        interpolation=cv2.INTER_LINEAR,
    )

    if mask is not None:
        mask = (mask * 255).astype(np.uint8)
        mask = ndimage.gaussian_filter(mask, sigma=calibration_blur_sigma)
        mask = cv2.resize(
            mask,
            None,
            fx=calibration_downsampling_factor,
            fy=calibration_downsampling_factor,
            interpolation=cv2.INTER_LINEAR,
        )

        img = np.ma.masked_where(mask > 127, img)

    return img


def condition_disparity(disp, eps=1e-6):
    disp = ndimage.median_filter(disp, size=3)
    disp = disp - np.min(disp)
    disp = disp / np.std(disp)
    disp = np.clip(disp, eps, np.inf)
    return disp


def imread(path, *args, **kwargs):
    img = cv2.imread(path, *args, **kwargs)
    if img is None:
        raise RuntimeError(f"Failed to read image from path '{path}'")
    return img