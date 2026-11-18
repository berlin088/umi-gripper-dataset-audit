"""Draft report generated from results."""


def write_report(out, sum_df, exact, near):
    with open(out / "audit_report_draft.md", "w") as f:
        f.write("# Dataset audit - auto-generated draft\n\n## Per-video summary\n\n")
        f.write(sum_df.round(2).to_markdown(index=False))
        f.write(f"\n\nExact duplicate files: {len(exact)}  |  Near-duplicate pairs: {len(near)}\n")
        f.write("\n## Automatic flags (you decide: confirmed defect vs potential risk)\n")
        for _, r in sum_df.iterrows():
            f.write(f"- **{r.video}**: {r.flags or 'none'}\n")
