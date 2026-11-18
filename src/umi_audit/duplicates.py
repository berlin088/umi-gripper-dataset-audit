"""Exact and near-duplicate detection, naming consistency."""
import itertools
import numpy as np, pandas as pd


def exact_duplicates(meta_df):
    return meta_df[meta_df.duplicated("sha256", keep=False)][["video", "sha256"]]


def near_duplicates(hashes, thresh):
    out = []
    for a, b in itertools.combinations(hashes, 2):
        ha, hb = hashes[a], hashes[b]
        m = min(len(ha), len(hb))
        if m:
            d = float(np.mean([ha[k] - hb[k] for k in range(m)]))
            if d < thresh:
                out.append(dict(a=a, b=b, mean_phash_distance=d))
    return pd.DataFrame(out, columns=["a", "b", "mean_phash_distance"])


def naming_check(meta_df):
    v = meta_df.video
    return pd.DataFrame(dict(video=v, has_parentheses=v.str.contains(r"[()]"),
                             has_uppercase=v.str.contains(r"[A-Z]"), has_spaces=v.str.contains(" ")))
