from typing import Optional
from collections import OrderedDict
from dataclasses import dataclass
import logging
import os
import glob
import math
import csv
import numpy as np
import cv2
from config import Config
from dpt import DPT
from megadetector import MegaDetector, MegaDetectorLabel
from deepmac import DeepMac
from custom_types import DetectionSamplingMethod, MultipleAnimalReduction, SampleFrom
from utils import calibrate, crop, resize, exception_to_str, get_calibration_frame_dist, get_extension_agnostic_path, multi_file_extension_glob
from visualization import visualize_detection, visualize_farthest_calibration_frame


@dataclass
class StatusUpdate():
    current_transect_id: str
    current_transect_idx: int
    total_transects: int
    current_detection_id: Optional[str] = None
    current_detection_idx: Optional[str] = None
    total_detections: Optional[str] = None


def run(config: Config):

    assert os.path.isdir(config.data_dir), "Data dir is not a directory"
    assert os.path.isdir(os.path.join(config.data_dir, "transects")) and os.path.isdir(os.path.join(config.data_dir, "results")), "Data dir must contain 'transect' and 'results' subdirectories. Please consult the manual for the correct directory structure."
    assert len(glob.glob(os.path.join(config.data_dir, "transects", "*/"))), "The 'transect' subdirectory must contain at least one transect. Please consult the manual for the correct directory structure."

    yield
    dpt = DPT()
    yield
    megadetector = MegaDetector()
    yield
    if config.detection_sampling_method == DetectionSamplingMethod.DEEPMAC:
        deepmac = DeepMac()
        yield

    with open(os.path.join(config.data_dir, "results", "results.csv"), "w", newline="") as result_csv_file, open(os.path.join(config.data_dir, "results", "results.txt"), "w") as result_distance_file: 
        head_row_csv = ["transect_id", "frame_id", "detection_idx", "detection_confidence", "depth", "world_x", "world_y", "world_z"]
        head_row_txt = ["Camera trap*Label", "Observation*Radial distance"]
        result_csv_writer = csv.writer(result_csv_file) 
        result_csv_writer.writerow(head_row_csv)
        result_distance_file.write("\t".join(head_row_txt) + os.linesep)

        transect_dirs = sorted(glob.glob(os.path.join(config.data_dir, "transects", "*/")))
        for transect_idx, transect_dir in enumerate(transect_dirs):
            transect_id = os.path.basename(os.path.normpath(transect_dir))

            yield StatusUpdate(transect_id, transect_idx, len(transect_dirs))

            exp = -1 if config.calibrate_metric else 1
            calibration_frames = {}

            for calibration_frame_filename in (
                multi_file_extension_glob(os.path.join(transect_dir, "calibration_frames", "*"), config.intensity_image_extensions) +
                multi_file_extension_glob(os.path.join(transect_dir, "calibration_frames_cropped", "*"), config.intensity_image_extensions)  # for backwards compability. use crop configuration instead
            ):
                yield
                calibration_frame_id = os.path.splitext(
                    os.path.basename(calibration_frame_filename)
                )[0]
                dist = get_calibration_frame_dist(transect_dir, calibration_frame_id)
                img = crop(
                    cv2.imread(calibration_frame_filename),
                    config.crop_top, config.crop_bottom, config.crop_left, config.crop_right,
                )
                mask = crop(
                    cv2.imread(
                        get_extension_agnostic_path(
                            os.path.join(
                                transect_dir,
                                "calibration_frames_masks",
                                calibration_frame_id,
                            ),
                            config.intensity_image_extensions,
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    > 127,
                    config.crop_top, config.crop_bottom, config.crop_left, config.crop_right,
                )
                disp = dpt(img)
                disp = np.ma.masked_where(mask, disp)
                calibration_frames[dist] = disp

            yield

            # sort calibration frames
            calibration_frames = OrderedDict(sorted(calibration_frames.items(), key=lambda kv: kv[0]))

            # get disparity of the farthest calibration frame
            farthest_calibration_frame_disp = list(calibration_frames.values())[-1] if len(calibration_frames) > 0 else None

            try:
                x,y  = [], []
                for dist, disp in calibration_frames.items():
                    yield
                    disp = resize(disp, farthest_calibration_frame_disp.shape)
                    disp_calibrated = calibrate(
                        disp ** exp,
                        farthest_calibration_frame_disp ** exp,
                        config.calibration_regression_method,
                    )(disp.data ** exp) ** exp
                    disp_calibrated = np.ma.masked_where(disp.mask, disp_calibrated)

                    x.append(np.median(disp_calibrated.data[disp_calibrated.mask]))
                    y.append(dist ** -1)

                calibration = calibrate(np.array(x) ** exp, np.array(y) ** exp, config.calibration_regression_method)
                farthest_calibration_frame_disp = np.ma.masked_where(
                    farthest_calibration_frame_disp.mask,
                    calibration(farthest_calibration_frame_disp.data ** exp) ** exp,
                )
            except Exception as e:
                calibration = None
                farthest_calibration_frame_disp = None
                if not os.path.exists(os.path.join(transect_dir, "detection_frames_depth")):
                    logging.warn(f"Failed calibrating transect '{transect_id}' due to exception: {exception_to_str(e)}. Skipping all distance estimations for observations in this transect.")

            yield

            if config.make_figures and farthest_calibration_frame_disp is not None:
                visualize_farthest_calibration_frame(config.data_dir, transect_id, farthest_calibration_frame_disp, config.min_depth, config.max_depth)

            detection_frame_filenames = sorted(list(set(
                multi_file_extension_glob(os.path.join(transect_dir, "detection_frames", "*"), config.intensity_image_extensions) +
                multi_file_extension_glob(os.path.join(transect_dir, "detection_frames_cropped", "*"), config.intensity_image_extensions)  # for backwards compability. use crop configuration instead
            )))
            for detection_idx, detection_frame_filename in enumerate(detection_frame_filenames):
                detection_id = os.path.splitext(os.path.basename(detection_frame_filename))[0]
                yield StatusUpdate(transect_id, transect_idx, len(transect_dirs), detection_id, detection_idx, len(detection_frame_filenames))

                # load intensity image
                img = cv2.imread(detection_frame_filename)

                # crop and resize intensity image to have the same size as the reference images
                img = crop(
                    img,
                    config.crop_top, config.crop_bottom, config.crop_left, config.crop_right,
                )
                img = resize(img, farthest_calibration_frame_disp.shape)

                yield

                # check if depth from stereo camera exists or calibration succeeded
                precomputed_depth_filename = get_extension_agnostic_path(os.path.join(transect_dir, "detection_frames_depth", detection_id), config.depth_image_extensions)
                if precomputed_depth_filename is None and farthest_calibration_frame_disp is None:
                    logging.warn(f"Unable to perform distance estimation on detection '{detection_id}' due to failed calibration and no precomputed depth maps.")
                    continue
                elif precomputed_depth_filename is not None:
                    assert config.sample_from == SampleFrom.DETECTION, "Config must be set to sample from detection if using precomputed depth maps"
                    depth = cv2.imread(precomputed_depth_filename, cv2.IMREAD_UNCHANGED)
                    disp = np.clip(depth, config.min_depth, config.max_depth) ** -1
                elif precomputed_depth_filename is None and farthest_calibration_frame_disp is not None:
                    if config.sample_from == SampleFrom.DETECTION:
                        disp = dpt(img)
                        disp = calibrate(disp ** exp, farthest_calibration_frame_disp ** exp, config.calibration_regression_method)(disp ** exp) ** exp
                    elif config.sample_from == SampleFrom.REFERENCE:
                        disp = farthest_calibration_frame_disp
                    else:
                        raise RuntimeError(f"Invalid configuration value '{config.sample_from}' for configuration sample_from")
                    depth = np.clip(disp, config.max_depth ** -1, config.min_depth ** -1) ** -1

                yield

                # run animal detection
                scores, labels, boxes = megadetector(img)

                yield

                # discard all non-animal detections
                if config.detect_humans:
                    correct_label_idx = np.nonzero((labels.flatten() == MegaDetectorLabel.ANIMAL) | (labels.flatten() == MegaDetectorLabel.PERSON))
                else:
                    correct_label_idx = np.nonzero(labels.flatten() == MegaDetectorLabel.ANIMAL)
                scores, labels, boxes = scores[correct_label_idx], labels[correct_label_idx], boxes[correct_label_idx]

                # discard all detections with low confidence
                high_confidence_idx = np.nonzero(scores.flatten() >= config.bbox_confidence_threshold)
                scores, labels, boxes = scores[high_confidence_idx], labels[high_confidence_idx], boxes[high_confidence_idx]

                # sort from image center outwards
                centerness = [((img.shape[1] / 2) - (box[0] + box[2] / 2)) ** 2 + ((img.shape[0] / 2) - (box[1] + box[3] / 2)) ** 2 for box in boxes]
                centerness_idx = np.argsort(centerness)
                scores, labels, boxes = scores[centerness_idx], labels[centerness_idx], boxes[centerness_idx]

                if config.detection_sampling_method == DetectionSamplingMethod.DEEPMAC:
                    # compute DeepMAC masks
                    masks = deepmac(img, boxes)

                    yield
                else:
                    # dummy masks
                    masks = [None for _ in boxes]

                sampled_depths = []
                sample_locations = []
                world_positions = []
                for box, mask in zip(boxes, masks):
                    yield
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    if config.detection_sampling_method == DetectionSamplingMethod.BBOX_BOTTOM:
                        sample_location = (
                            max(0, min(depth.shape[0] - 1, round(box[3]))),
                            max(0, min(depth.shape[1] - 1, round(box[0] + (box[2] - box[0]) / 2))),
                        )
                        sampled_depths += [depth[sample_location]]
                        sample_locations += [sample_location]
                    elif config.detection_sampling_method == DetectionSamplingMethod.BBOX_PERCENTILE:
                        ymin, ymax = max(0, min(depth.shape[0] - 2, round(box[1]))), max(0, min(depth.shape[0] - 1, round(box[3])))
                        xmin, xmax = max(0, min(depth.shape[1] - 2, round(box[0]))), max(0, min(depth.shape[1] - 1, round(box[2])))
                        depth_cropped = depth[ymin:ymax, xmin:xmax]
                        sampled_depths += [np.percentile(depth_cropped, config.bbox_sampling_percentile, method="nearest")]
                        sample_location = np.nonzero(depth_cropped == sampled_depths[-1])
                        sample_location = (
                            round(sample_location[0][0] + box[1]),
                            round(sample_location[1][0] + box[0]),
                        )
                        sample_locations += [sample_location]
                    elif config.detection_sampling_method == DetectionSamplingMethod.DEEPMAC:
                        ymin, ymax = max(0, min(depth.shape[0] - 2, round(box[1]))), max(0, min(depth.shape[0] - 1, round(box[3])))
                        xmin, xmax = max(0, min(depth.shape[1] - 2, round(box[0]))), max(0, min(depth.shape[1] - 1, round(box[2])))
                        depth_cropped = depth[ymin:ymax, xmin:xmax]
                        mask_resized = cv2.resize(mask, (xmax - xmin, ymax - ymin), interpolation=cv2.INTER_LINEAR)
                        if np.sum(mask_resized >= 0.5) > 0:
                            sampled_depths += [np.percentile(depth_cropped[mask_resized >= 0.5], 50, method="nearest")]
                        else:
                            sampled_depths += [np.percentile(depth_cropped, config.bbox_sampling_percentile, method="nearest")]
                        sample_location = np.nonzero(depth_cropped == sampled_depths[-1])
                        sample_location = (
                            round(sample_location[0][0] + box[1]),
                            round(sample_location[1][0] + box[0]),
                        )
                        sample_locations += [sample_location]
                    else:
                        raise RuntimeError(f"Invalid configuration value '{config.detection_sampling_method}' for configuration detection_sampling_method")
    
                    # compute horizontal angle a
                    f = (0.5 * depth.shape[1]) / math.tan(0.5 * math.pi * config.camera_horizontal_fov / 180)
                    c = np.array([0, 0, f])
                    p = np.array([0, box[0] + box[2] / 2 - depth.shape[1] / 2, f])
                    a = math.copysign(1, box[0] + box[2] / 2 - depth.shape[1] / 2) * math.acos(c @ p / (np.linalg.norm(c) * np.linalg.norm(p)))

                    # compute vertical angle b
                    f = (0.5 * depth.shape[0]) / math.tan(0.5 * math.pi * config.camera_vertical_fov / 180)
                    c = np.array([0, 0, f])
                    p = np.array([box[1] + box[3] / 2 - depth.shape[0] / 2, 0, f])
                    b = math.copysign(1, box[1] + box[3] / 2 - depth.shape[0] / 2) * math.acos(c @ p / (np.linalg.norm(c) * np.linalg.norm(p)))

                    # compute world position
                    x = sampled_depths[-1] * math.tan(a)
                    y = sampled_depths[-1] * math.tan(b)
                    z = sampled_depths[-1] * math.cos(a) * math.cos(b)
                    world_positions += [[x, y, z]]

                    if config.multiple_animal_reduction == MultipleAnimalReduction.ONLY_CENTERMOST:
                        break

                if config.multiple_animal_reduction == MultipleAnimalReduction.MEDIAN:
                    sampled_depths = [np.median(sampled_depths)] if len(sampled_depths) > 0 else []
                    world_positions = [np.mean(world_positions, axis=0)]


                if config.make_figures:
                    visualize_detection(config.data_dir, detection_id, img, depth, farthest_calibration_frame_disp, boxes, masks, world_positions, sample_locations, config.draw_world_position, config.min_depth, config.max_depth)

                yield

                for i, (score, sampled_depth, world_position) in enumerate(zip(scores, sampled_depths, world_positions)):
                    detection_i = i if config.multiple_animal_reduction != MultipleAnimalReduction.MEDIAN else -1
                    result_csv_writer.writerow([transect_id, detection_id, f"{detection_i:03d}", f"{score.item():.4f}", f"{sampled_depth.item():.4f}", f"{world_position[0].item():.4f}", f"{world_position[1].item():.4f}", f"{world_position[2].item():.4f}"])
                    result_distance_file.write("\t".join([transect_id, f"{sampled_depth.item():.4f}"]) + os.linesep)