"""Orchestrates the audit:  umi-audit --videos data/raw"""
import argparse
from pathlib import Path
import cv2, numpy as np, pandas as pd, imagehash
from PIL import Image

from .config import Config, VIDEO_EXT
from .video_io import sha256, ffprobe, frac
from .roi import resolve_roi
from .markers import make_detector, detect_rows
from .metrics import frame_metrics, summarise_video, flags
from .duplicates import exact_duplicates, near_duplicates, naming_check
from .plots import make_plots
from .report import write_report


def analyse_video(vp, cfg, detector, samples_dir):
    meta = ffprobe(vp)
    cap = cv2.VideoCapture(str(vp))
    W, H = int(cap.get(3)), int(cap.get(4))
    roi = resolve_roi(cap, cfg)
    if not meta:
        print("  WARNING: ffprobe returned nothing (is ffmpeg on PATH?). Falling back to OpenCV metadata.")
    fps = frac(meta.get("avg_frame_rate", "0/1"))
    if not fps or np.isnan(fps):
        fps = cap.get(cv2.CAP_PROP_FPS)
    frame_rows, marker_rows, hashes, prev, i, mid_saved = [], [], [], None, 0, False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % cfg.step == 0:
            if roi:
                x, y, w, h = roi
                fr = fr[y:y + h, x:x + w]
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            rows, corners, ids = detect_rows(detector, g, vp.name, i)
            marker_rows += rows
            frame_rows.append(dict(video=vp.name, frame=i, t=i / fps if fps else np.nan,
                                   n_markers=len(rows), **frame_metrics(g, prev)))
            prev = g
            if (i // cfg.step) % 10 == 0:
                hashes.append(imagehash.phash(Image.fromarray(g)))
            if not mid_saved and i >= total // 2:        # evidence frame with detections drawn
                im = fr.copy()
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(im, corners, ids)
                cv2.imwrite(str(samples_dir / f"{vp.stem}_mid.jpg"), im)
                mid_saved = True
        i += 1
    cap.release()
    meta_row = dict(video=vp.name, size_mb=round(vp.stat().st_size / 1e6, 2), sha256=sha256(vp),
                    codec=meta.get("codec_name"), pix_fmt=meta.get("pix_fmt"), width=W, height=H,
                    aspect=round(W / H, 3), roi=str(roi) if roi else "full frame",
                    fps_avg=fps, fps_nominal=frac(meta.get("r_frame_rate", "0/1")),
                    duration_s=float(meta.get("duration") or np.nan),
                    bitrate_mbps=round(float(meta.get("bit_rate") or 0) / 1e6, 2),
                    frames_meta=meta.get("nb_frames"), frames_decoded=i)
    return meta_row, pd.DataFrame(frame_rows), pd.DataFrame(marker_rows), hashes


def run(cfg):
    out = Path(cfg.out); (out / "samples").mkdir(parents=True, exist_ok=True)
    vids = sorted(p for p in Path(cfg.videos).iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not vids:
        raise SystemExit(f"No videos found in {cfg.videos}")
    print(f"{len(vids)} videos found")
    detector = make_detector()
    metas, frames, markers, sums, hashes = [], [], [], [], {}
    failed = []
    for vp in vids:
        print("analysing", vp.name)
        try:
            m, fr, mk, ph = analyse_video(vp, cfg, detector, out / "samples")
        except Exception as e:
            print(f"  ERROR on {vp.name}: {e}")
            failed.append(dict(video=vp.name, error=str(e)))
            continue
        s = summarise_video(m, fr, mk, cfg); s["flags"] = flags(m, s)
        metas.append(m); frames.append(fr); markers.append(mk); sums.append(s); hashes[vp.name] = ph
    meta_df, sum_df = pd.DataFrame(metas), pd.DataFrame(sums)
    fr_df = pd.concat(frames, ignore_index=True)
    non_empty = [m for m in markers if len(m)]
    mk_df = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    exact, near, names = exact_duplicates(meta_df), near_duplicates(hashes, cfg.dup_phash_dist), naming_check(meta_df)

    meta_df.to_csv(out / "video_metadata.csv", index=False)
    sum_df.to_csv(out / "quality_summary.csv", index=False)
    fr_df.to_csv(out / "frame_metrics.csv", index=False)
    if len(mk_df): mk_df.to_csv(out / "marker_detections.csv", index=False)
    exact.to_csv(out / "exact_duplicates.csv", index=False)
    near.to_csv(out / "near_duplicates.csv", index=False)
    names.to_csv(out / "naming_check.csv", index=False)
    pd.DataFrame(failed).to_csv(out / "failed_videos.csv", index=False)
    make_plots(fr_df, mk_df, out, len(vids))
    write_report(out, sum_df, exact, near)
    print("\nDone ->", out.resolve())
    return sum_df


def main():
    ap = argparse.ArgumentParser(description="UMI gripper dataset audit")
    d = Config()
    ap.add_argument("--videos", default=d.videos); ap.add_argument("--out", default=d.out)
    ap.add_argument("--roi", default=None, help="x,y,w,h manual crop")
    ap.add_argument("--no-roi", action="store_true", help="use full frame (original files)")
    ap.add_argument("--trim-bottom", type=float, default=d.trim_bottom)
    ap.add_argument("--step", type=int, default=d.step)
    ap.add_argument("--blur-thresh", type=float, default=d.blur_thresh)
    a = ap.parse_args()
    cfg = Config(videos=a.videos, out=a.out, roi=a.roi, no_roi=a.no_roi, trim_bottom=a.trim_bottom,
                 step=a.step, blur_thresh=a.blur_thresh)
    print(run(cfg).round(2).T.to_string())


if __name__ == "__main__":
    main()
