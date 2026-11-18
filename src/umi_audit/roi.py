"""Find the video player area inside a screen recording."""
import cv2, numpy as np


def auto_roi(cap, n_samples=40):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(total - 1, 0), n_samples).astype(int)
    stack = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            small = cv2.resize(fr, None, fx=0.25, fy=0.25)
            stack.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if len(stack) < 5:
        return None
    sd = np.std(np.stack(stack), axis=0)
    mask = (sd > 12).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    if w * h < 0.1 * mask.size:
        return None
    return (x * 4, y * 4, w * 4, h * 4)


def resolve_roi(cap, cfg):
    if cfg.no_roi:
        return None
    if cfg.roi:
        return tuple(int(v) for v in cfg.roi.split(","))
    roi = auto_roi(cap)
    if roi and cfg.trim_bottom > 0:
        x, y, w, h = roi
        roi = (x, y, w, int(h * (1 - cfg.trim_bottom)))
    return roi
