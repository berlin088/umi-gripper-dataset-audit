"""Figures for the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _box(df, col, title, out, n):
    df.boxplot(column=col, by="video", rot=90, figsize=(max(6, n * .8), 5))
    plt.title(title); plt.suptitle(""); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def make_plots(fr_df, mk_df, out, n_videos):
    for col, title in [("blur", "Sharpness (Laplacian variance)"), ("brightness", "Mean brightness"),
                       ("n_markers", "Markers detected per frame")]:
        _box(fr_df, col, title, out / f"box_{col}.png", n_videos)
    if len(mk_df):
        _box(mk_df, "side_px", "Marker side length (px)", out / "box_marker_size.png", n_videos)
        fig, ax = plt.subplots(figsize=(8, 4))
        mk_df.id.value_counts().sort_index().plot.bar(ax=ax)
        ax.set_title("Marker ID frequency (all videos)")
        plt.tight_layout(); plt.savefig(out / "marker_id_freq.png", dpi=150); plt.close()
    fig, axes = plt.subplots(n_videos, 1, figsize=(10, 1.6 * n_videos), squeeze=False)
    for ax, (name, g) in zip(axes[:, 0], fr_df.groupby("video")):
        ax.fill_between(g.t, 0, (g.n_markers > 0).astype(int), step="mid")
        ax.set_ylabel(name[:14], fontsize=6); ax.set_yticks([])
    axes[-1, 0].set_xlabel("time (s) - filled = at least one marker detected")
    plt.tight_layout(); plt.savefig(out / "marker_visibility_timeline.png", dpi=150); plt.close()
