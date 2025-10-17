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
from dpt_pytorch import DPTPyTorch
from depth_anything import DepthAnything
from metric3d import Metric3D
from megadetector import MegaDetector, MegaDetectorLabel
from sam import SAM
from custom_types import DetectionSamplingMethod, MultipleAnimalReduction, SampleFrom, DepthEstimationModel
from utils import calibrate, calibrate_v0, crop, resize, exception_to_str, get_calibration_frame_dist, get_extension_agnostic_path, multi_file_extension_glob, blur_and_downsample, imread
from visualization import visualize_detection, visualize_farthest_calibration_frame


@dataclass
class StatusUpdate():
    current_transect_id: str
    current_transect_idx: int
    total_transects: int
    current_detection_id: Optional[str] = None
    current_detection_idx: Optional[str] = None
    total_detections: Optional[str] = None


def run(config: Config, gui=False):

    eps = 1e-6

    assert os.path.isdir(config.data_dir), "Data dir is not a directory"
    assert os.path.isdir(os.path.join(config.data_dir, "transects")) and os.path.isdir(os.path.join(config.data_dir, "results")), "Data dir must contain 'transect' and 'results' subdirectories. Please consult the manual for the correct directory structure."
    assert len(glob.glob(os.path.join(config.data_dir, "transects", "*/"))), "The 'transect' subdirectory must contain at least one transect. Please consult the manual for the correct directory structure."
    # assert config.depth_estimation_model != DepthEstimationModel.METRIC_3D_V2_VIT_S or config.calibrate_metric == True

    yield

    if gui:
        try:
            from desktop_notifier import DesktopNotifier
            notifier = DesktopNotifier()
        except Exception as e:
            logging.error(f"Failed to initialize desktop notifier for GUI notifications: {exception_to_str(e)}")
            notifier = None

    do_calibrate = calibrate
    if config.depth_estimation_model == DepthEstimationModel.DPT:
        depth_estimation_model = DPT()
    elif config.depth_estimation_model == DepthEstimationModel.DPT_PYTORCH:
        depth_estimation_model = DPTPyTorch()
        do_calibrate = calibrate_v0
    elif config.depth_estimation_model == DepthEstimationModel.DEPTH_AHYTHING_METRIC:
        depth_estimation_model = DepthAnything()
    elif config.depth_estimation_model == DepthEstimationModel.METRIC_3D_V2_VIT_S:
        depth_estimation_model = Metric3D()
    else:
        raise ValueError(f"Invalud depth estimation model '{config.depth_estimation_model}'")
    yield
    megadetector = MegaDetector()
    yield
    if config.detection_sampling_method == DetectionSamplingMethod.SAM:
        sam = SAM()
        yield

    with open(os.path.join(config.data_dir, "results", "results.csv"), "w", newline="") as result_csv_file, open(os.path.join(config.data_dir, "results", "results.txt"), "w") as result_distance_file: 
        head_row_csv = ["transect_id", "frame_id", "detection_idx", "detection_confidence", "depth", "world_x", "world_y", "world_z", "error_status"]
        head_row_txt = ["Camera trap*Label", "Observation*Radial distance"]
        result_csv_writer = csv.writer(result_csv_file) 
        result_csv_writer.writerow(head_row_csv)
        result_distance_file.write("\t".join(head_row_txt) + os.linesep)

        transect_dirs = sorted(glob.glob(os.path.join(config.data_dir, "transects", "*/")))
        for transect_idx, transect_dir in enumerate(transect_dirs):
            transect_id = os.path.basename(os.path.normpath(transect_dir))

            yield StatusUpdate(transect_id, transect_idx, len(transect_dirs))

            try:
                exp = -1 if config.calibrate_metric else 1
                calibration_frames = {}
                farthest_calibration_frame_disp = None

                if config.depth_estimation_model != DepthEstimationModel.DEPTH_AHYTHING_METRIC:
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
                                imread(calibration_frame_filename),
                                config.crop_top, config.crop_bottom, config.crop_left, config.crop_right,
                            )
                            mask = crop(
                                imread(
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
                            disp = depth_estimation_model(img)
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
                            if config.calibrate_metric:
                                disp = np.clip(disp, eps, np.inf)
                            disp_calibrated = do_calibrate(
                                disp ** exp,
                                farthest_calibration_frame_disp ** exp,
                                config.calibration_regression_method,
                            )(disp.data ** exp) ** exp
                            disp_calibrated = np.ma.masked_where(disp.mask, disp_calibrated)

                            x.append(np.median(disp_calibrated.data[disp_calibrated.mask]))
                            y.append(dist ** -1)

                        calibration = do_calibrate(np.array(x) ** exp, np.array(y) ** exp, config.calibration_regression_method)
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

            except Exception as e:
                exception_str = exception_to_str(e)
                log_str = f"Error processing transect '{transect_id}': {exception_str}"
                notification_str = f"Error processing transect '{transect_id}': {str(e)}"
                logging.error(log_str)

                if gui and notifier is not None:
                    try:
                        notifier.send(title="Distance Estimation Error", message=notification_str)
                    except Exception as e:
                        logging.error(f"Failed to send desktop notification: {exception_to_str(e)}")
                result_csv_writer.writerow([transect_id, "", "", "", "", "", "", "", notification_str])
            
            detection_frame_filenames = sorted(list(set(
                multi_file_extension_glob(os.path.join(transect_dir, "detection_frames", "*"), config.intensity_image_extensions) +
                multi_file_extension_glob(os.path.join(transect_dir, "detection_frames_cropped", "*"), config.intensity_image_extensions)  # for backwards compability. use crop configuration instead
            )))
            for detection_idx, detection_frame_filename in enumerate(detection_frame_filenames):
                detection_id = os.path.splitext(os.path.basename(detection_frame_filename))[0]
                yield StatusUpdate(transect_id, transect_idx, len(transect_dirs), detection_id, detection_idx, len(detection_frame_filenames))

                try:

                    # load intensity image
                    img = imread(detection_frame_filename)

                    # crop and resize intensity image to have the same size as the reference images
                    img = crop(
                        img,
                        config.crop_top, config.crop_bottom, config.crop_left, config.crop_right,
                    )
                    if farthest_calibration_frame_disp is not None:
                        img = resize(img, farthest_calibration_frame_disp.shape)

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
                    centerness = [((img.shape[1] / 2) - (box[0] + box[2]) / 2) ** 2 + ((img.shape[0] / 2) - (box[1] + box[3]) / 2) ** 2 for box in boxes]
                    centerness_idx = np.argsort(centerness)
                    scores, labels, boxes = scores[centerness_idx], labels[centerness_idx], boxes[centerness_idx]

                    if config.detection_sampling_method == DetectionSamplingMethod.SAM:
                        # compute SAM masks
                        masks = sam(img, boxes)
                        animal_mask = np.any(masks, axis=0)

                        yield
                    else:
                        # dummy masks
                        masks = [None for _ in boxes]

                        # compute animal mask from bounding boxes
                        animal_mask = np.zeros(img.shape[0:2], dtype=bool)
                        for box in boxes:
                            ymin, ymax = max(0, min(img.shape[0] - 2, round(box[1]))), max(0, min(img.shape[0] - 1, round(box[3])))
                            xmin, xmax = max(0, min(img.shape[1] - 2, round(box[0]))), max(0, min(img.shape[1] - 1, round(box[2])))
                            animal_mask[ymin:ymax, xmin:xmax] = True


                    # check if using metric depth model
                    if config.depth_estimation_model == DepthEstimationModel.DEPTH_AHYTHING_METRIC:
                        assert config.sample_from == SampleFrom.DETECTION, "Config must be set to sample from detection if using metric depth model"
                        depth = depth_estimation_model(img)
                        disp = np.clip(depth, config.min_depth, config.max_depth) ** -1
                    else:
                        # check if depth from stereo camera exists or calibration succeeded
                        precomputed_depth_filename = get_extension_agnostic_path(os.path.join(transect_dir, "detection_frames_depth", detection_id), config.depth_image_extensions)
                        if precomputed_depth_filename is None and farthest_calibration_frame_disp is None:
                            logging.warn(f"Unable to perform distance estimation on detection '{detection_id}' due to failed calibration and no precomputed depth maps.")
                            continue
                        elif precomputed_depth_filename is not None:
                            assert config.sample_from == SampleFrom.DETECTION, "Config must be set to sample from detection if using precomputed depth maps"
                            depth = imread(precomputed_depth_filename, cv2.IMREAD_UNCHANGED)
                            disp = np.clip(depth, config.min_depth, config.max_depth) ** -1
                        elif precomputed_depth_filename is None and farthest_calibration_frame_disp is not None:
                            if config.sample_from == SampleFrom.DETECTION:
                                disp = depth_estimation_model(img)
                                if config.calibrate_metric:
                                    disp = np.clip(disp, eps, np.inf)
                                mask = (farthest_calibration_frame_disp ** -1) >= (config.max_depth - eps)
                                if config.calibration_mask_animals:
                                    mask = mask | animal_mask
                                disp_masked = np.ma.masked_where(mask, disp ** exp)
                                if config.calibrate_blur:
                                    disp = do_calibrate(blur_and_downsample(disp_masked), blur_and_downsample(farthest_calibration_frame_disp) ** exp, config.calibration_regression_method)(disp ** exp)
                                else:
                                    disp = do_calibrate(disp_masked, farthest_calibration_frame_disp ** exp, config.calibration_regression_method)(disp ** exp)
                                if config.calibrate_metric:
                                    disp = np.clip(disp, config.min_depth, config.max_depth) ** -1
                            elif config.sample_from == SampleFrom.REFERENCE:
                                disp = farthest_calibration_frame_disp
                            else:
                                raise RuntimeError(f"Invalid configuration value '{config.sample_from}' for configuration sample_from")
                            depth = np.clip(disp, config.max_depth ** -1, config.min_depth ** -1) ** -1

                    yield

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
                        elif config.detection_sampling_method == DetectionSamplingMethod.SAM:
                            ymin, ymax = max(0, min(depth.shape[0] - 2, round(box[1]))), max(0, min(depth.shape[0] - 1, round(box[3])))
                            xmin, xmax = max(0, min(depth.shape[1] - 2, round(box[0]))), max(0, min(depth.shape[1] - 1, round(box[2])))
                            depth_cropped = depth[ymin:ymax, xmin:xmax]
                            mask_cropped = mask[ymin:ymax, xmin:xmax]
                            mask_paddded = np.pad(mask_cropped, ((1, 1), (1, 1)))
                            dist = cv2.distanceTransform((mask_paddded * 255).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_3)
                            sample_location = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
                            sample_location = (
                                max(0, min(mask_cropped.shape[0], sample_location[0] - 1)),
                                max(0, min(mask_cropped.shape[1], sample_location[1] - 1)),
                            )
                            sampled_depths += [depth_cropped[sample_location[0], sample_location[1]]]
                            sample_location = (
                                round(sample_location[0] + box[1]),
                                round(sample_location[1] + box[0]),
                            )
                            sample_locations += [sample_location]
                        else:
                            raise RuntimeError(f"Invalid configuration value '{config.detection_sampling_method}' for configuration detection_sampling_method")
        
                        # compute horizontal angle a
                        f = (0.5 * depth.shape[1]) / math.tan(0.5 * math.pi * config.camera_horizontal_fov / 180)
                        c = np.array([0, 0, f])
                        p = np.array([(box[0] + box[2]) / 2 - depth.shape[1] / 2, 0, f])
                        a = math.copysign(1, (box[0] + box[2]) / 2 - depth.shape[1] / 2) * math.acos((c @ p) / (np.linalg.norm(c) * np.linalg.norm(p)))

                        # compute vertical angle b
                        f = (0.5 * depth.shape[0]) / math.tan(0.5 * math.pi * config.camera_vertical_fov / 180)
                        c = np.array([0, 0, f])
                        p = np.array([0, (box[1] + box[3]) / 2 - depth.shape[0] / 2, f])
                        b = math.copysign(1, (box[1] + box[3]) / 2 - depth.shape[0] / 2) * math.acos((c @ p) / (np.linalg.norm(c) * np.linalg.norm(p)))

                        # compute world position
                        z = sampled_depths[-1] / math.sqrt(math.tan(a) ** 2 + math.tan(b) ** 2 + 1)
                        x = z * math.tan(a)
                        y = z * math.tan(b)
                        world_positions += [[x, y, z]]

                        if config.multiple_animal_reduction == MultipleAnimalReduction.ONLY_CENTERMOST:
                            break

                    if config.multiple_animal_reduction == MultipleAnimalReduction.MEDIAN:
                        sampled_depths = [np.median(sampled_depths)] if len(sampled_depths) > 0 else []
                        world_positions = [np.mean(world_positions, axis=0)]


                    if config.make_figures:
                        visualize_detection(config.data_dir, detection_id, img, depth, farthest_calibration_frame_disp, boxes, masks, world_positions, sample_locations, config.draw_detection_ids, config.draw_world_position, config.min_depth, config.max_depth)

                    yield

                    for i, (score, sampled_depth, world_position) in enumerate(zip(scores, sampled_depths, world_positions)):
                        detection_i = i if config.multiple_animal_reduction != MultipleAnimalReduction.MEDIAN else -1
                        result_csv_writer.writerow([transect_id, detection_id, f"{detection_i:03d}", f"{score.item():.4f}", f"{sampled_depth.item():.4f}", f"{world_position[0].item():.4f}", f"{world_position[1].item():.4f}", f"{world_position[2].item():.4f}", ""])
                        result_distance_file.write("\t".join([transect_id, f"{sampled_depth.item():.4f}"]) + os.linesep)

                except Exception as e:
                    exception_str = exception_to_str(e)
                    log_str = f"Error processing detection '{detection_id}' in transect '{transect_id}': {exception_str}"
                    notification_str = f"Error processing detection '{detection_id}' in transect '{transect_id}': {str(e)}"
                    logging.error(log_str)

                    if gui and notifier is not None:
                        try:
                            notifier.send(title="Distance Estimation Error", message=notification_str)
                        except Exception as e:
                            logging.error(f"Failed to send desktop notification: {exception_to_str(e)}")
                    result_csv_writer.writerow([transect_id, "", "", "", "", "", "", "", notification_str])
