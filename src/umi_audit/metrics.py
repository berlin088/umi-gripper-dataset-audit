"""Per-frame image quality metrics and per-video summaries."""
import numpy as np, cv2
from .config import ADVERTISED_RES, ADVERTISED_FPS


def frame_metrics(gray, prev_gray):
    ok = prev_gray is not None and prev_gray.shape == gray.shape
    mad = float(np.mean(cv2.absdiff(gray, prev_gray))) if ok else np.nan
    return dict(blur=cv2.Laplacian(gray, cv2.CV_64F).var(), brightness=float(gray.mean()),
                contrast=float(gray.std()), dark_frac=float((gray < 10).mean()),
                bright_frac=float((gray > 245).mean()), mad_prev=mad)


def summarise_video(meta, fr, mk, cfg):
    s = dict(video=meta["video"], frames_analysed=len(fr))
    s["blur_median"] = fr.blur.median()
    s["blur_p10"] = fr.blur.quantile(0.10)
    s["blurry_frame_pct"] = 100 * (fr.blur < cfg.blur_thresh).mean()
    s["brightness_mean"] = fr.brightness.mean()
    s["underexposed_pct"] = 100 * (fr.brightness < 50).mean()
    s["overexposed_pct"] = 100 * (fr.brightness > 200).mean()
    s["clipped_bright_px_mean_pct"] = 100 * fr.bright_frac.mean()
    s["frozen_frame_pct"] = 100 * (fr.mad_prev < cfg.frozen_mad).mean()
    s["frames_with_marker_pct"] = 100 * (fr.n_markers > 0).mean()
    s["frames_with_2plus_markers_pct"] = 100 * (fr.n_markers >= 2).mean()
    s["frames_no_marker_pct"] = 100 * (fr.n_markers == 0).mean()
    if len(mk):
        s["marker_ids_seen"] = ",".join(map(str, sorted(mk.id.unique())))
        s["marker_side_px_median"] = mk.side_px.median()
        s["marker_side_px_p10"] = mk.side_px.quantile(0.10)
        s["small_marker_pct"] = 100 * (mk.side_px < cfg.small_marker_px).mean()
        jit = []
        for _, g in mk.sort_values("frame").groupby("id"):
            if len(g) >= 15:
                for c in ("cx", "cy"):
                    jit.append(np.nanstd(g[c].values - g[c].rolling(5, center=True).mean().values))
        s["centre_jitter_px"] = float(np.nanmean(jit)) if jit else np.nan
        best = cur = 0
        for b in (fr.n_markers.values == 0):
            cur = cur + 1 if b else 0
            best = max(best, cur)
        s["longest_no_marker_gap_frames"] = best * cfg.step
        top = mk.id.value_counts().index[:2].tolist()
        if len(top) == 2:
            a = mk[mk.id == top[0]].set_index("frame"); b = mk[mk.id == top[1]].set_index("frame")
            common = a.index.intersection(b.index)
            if len(common) > 5:
                d = np.hypot(a.loc[common, "cx"] - b.loc[common, "cx"], a.loc[common, "cy"] - b.loc[common, "cy"]) \
                    / ((a.loc[common, "side_px"] + b.loc[common, "side_px"]) / 2)
                s.update(opening_proxy_pair=f"{top[0]}-{top[1]}", opening_proxy_min=float(d.min()),
                         opening_proxy_max=float(d.max()), opening_proxy_range=float(d.max() - d.min()))
    return s


def flags(meta, s):
    f = []
    if meta["roi"] != "full frame":
        f.append("SCREEN-RECORDING/ROI: resolution/fps/blur are not the original data's")
    elif (meta["width"], meta["height"]) != ADVERTISED_RES:
        f.append(f"resolution {meta['width']}x{meta['height']} != advertised {ADVERTISED_RES[0]}x{ADVERTISED_RES[1]}")
    if meta["roi"] == "full frame" and not (abs((meta["fps_avg"] or 0) - ADVERTISED_FPS) <= 0.05):
        f.append(f"fps {meta['fps_avg']:.2f} != {ADVERTISED_FPS}")
    if str(meta["frames_meta"]) not in ("None", str(meta["frames_decoded"])):
        f.append("frame count metadata != decoded")
    if s.get("frames_no_marker_pct", 0) > 30: f.append("markers missing in >30% of frames")
    if s.get("small_marker_pct", 0) > 20: f.append(">20% tiny marker detections")
    if s.get("frozen_frame_pct", 0) > 5: f.append(">5% frozen/duplicate frames")
    if s.get("blurry_frame_pct", 0) > 30: f.append(">30% blurry frames")
    if s.get("marker_ids_seen") and len(s["marker_ids_seen"].split(",")) > 4:
        f.append("many marker IDs: check which belong to the flaps")
    return "; ".join(f)
