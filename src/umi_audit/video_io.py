"""File-level checks: ffprobe metadata, hashing."""
import hashlib, json, subprocess
import numpy as np


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def frac(s):
    try:
        a, b = str(s).split("/")
        return float(a) / float(b)
    except Exception:
        return np.nan


def ffprobe(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,bit_rate,pix_fmt",
           "-of", "json", str(path)]
    try:
        return json.loads(subprocess.check_output(cmd))["streams"][0]
    except Exception:
        return {}
