"""All tunable settings in one place."""
from __future__ import annotations
from dataclasses import dataclass
import cv2

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
ARUCO_DICT = cv2.aruco.DICT_4X4_50     # confirmed on sample videos
ADVERTISED_RES = (3840, 3360)          # from the dataset web page
ADVERTISED_FPS = 29.97


@dataclass
class Config:
    videos: str = "data/raw"
    out: str = "outputs/audit"
    roi: str | None = None             # "x,y,w,h" manual crop
    no_roi: bool = False               # True for original downloaded files
    trim_bottom: float = 0.10          # cut player control bar from auto ROI
    step: int = 3                      # analyse every Nth frame
    blur_thresh: float = 60.0          # Laplacian variance below this = blurry
    small_marker_px: float = 20.0
    frozen_mad: float = 0.05           # mean abs diff below this = frozen frame
    dup_phash_dist: float = 8.0
