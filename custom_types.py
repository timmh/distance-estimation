from enum import Enum, auto


class DetectionSamplingMethod(Enum):
    BBOX_BOTTOM = auto()
    BBOX_PERCENTILE = auto()
    DEEPMAC = auto()


class SampleFrom(Enum):
    REFERENCE = auto()
    DETECTION = auto()


class MultipleAnimalReduction(Enum):
    NONE = auto()
    MEDIAN = auto()
    ONLY_CENTERMOST = auto()


class RegressionMethod(Enum):
    RANSAC = auto()
    LEASTSQUARES = auto()
    POLY = auto()
    RANSAC_POLY = auto()