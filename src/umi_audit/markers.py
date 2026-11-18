"""ArUco marker detection and geometry."""
import cv2, numpy as np
from .config import ARUCO_DICT


def make_detector(dictionary=ARUCO_DICT):
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary),
                                   cv2.aruco.DetectorParameters())


def marker_geometry(corners):
    c = corners.reshape(4, 2)
    centre = c.mean(0)
    side = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))
    return centre, side, c


def detect_rows(detector, gray, video, frame_idx):
    """Return (rows, corners, ids) for one grayscale frame."""
    corners, ids, _ = detector.detectMarkers(gray)
    rows = []
    if ids is None:
        return rows, corners, ids
    for cs, mid in zip(corners, ids.flatten()):
        centre, side, c = marker_geometry(cs)
        rows.append(dict(video=video, frame=frame_idx, id=int(mid), cx=centre[0], cy=centre[1],
                         side_px=side, **{f"{k}{j}": c[j][i] for j in range(4) for i, k in enumerate("xy")}))
    return rows, corners, ids
