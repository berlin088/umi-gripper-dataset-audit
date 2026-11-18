import numpy as np, cv2
from umi_audit.markers import make_detector, detect_rows
from umi_audit.metrics import frame_metrics


def test_detects_generated_marker():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(d, 7, 120)
    canvas = np.full((300, 300), 255, np.uint8)
    canvas[90:210, 90:210] = marker
    rows, _, _ = detect_rows(make_detector(), canvas, "t.mp4", 0)
    assert len(rows) == 1 and rows[0]["id"] == 7


def test_blur_lower_for_blurred_image():
    rng = np.random.default_rng(0)
    img = (rng.random((200, 200)) * 255).astype(np.uint8)
    sharp = frame_metrics(img, None)["blur"]
    soft = frame_metrics(cv2.GaussianBlur(img, (15, 15), 5), None)["blur"]
    assert soft < sharp
