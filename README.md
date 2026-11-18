# UMI Gripper Dataset Audit

Structured quality-control audit of the UMI gripper video dataset (fiducial markers on the two gripper flaps).
It measures file integrity, image quality, marker visibility, diversity and duplicates, and produces CSVs, figures
and a draft report. It does **not** implement pose estimation.

## Setup (VS Code)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```
`ffprobe` (part of ffmpeg) must be on PATH.

## Run
Put videos in `data/raw/` (never committed), then:
```bash
umi-audit --videos data/raw --out outputs/audit        # or: python scripts/run_audit.py ...
```
Options: `--no-roi` (original files), `--roi x,y,w,h` (manual crop), `--step 1` (every frame).

> If the videos are screen recordings of the website, the auto-ROI crops the player area. Resolution, fps and blur
> then describe the recording, not the dataset. State this in the report.

## Tests
```bash
pytest
```

## Layout
```
src/umi_audit/   config, video_io, roi, markers, metrics, duplicates, plots, report, pipeline
scripts/         launcher
tests/           smoke tests
docs/            report_template.md, findings_log.md
notebooks/       exploratory analysis of outputs/*.csv
data/raw/        videos (git-ignored)
outputs/         results (git-ignored)
```

## Outputs (`outputs/audit/`)
quality_summary.csv, video_metadata.csv, frame_metrics.csv, marker_detections.csv, duplicate and naming CSVs,
box plots, marker timeline, samples/ evidence frames, audit_report_draft.md.
