# 90_report.py
# ------------
# PURPOSE
# -------
# Generates lightweight, review-friendly reports from the final dataset
# (after sentiment + stance + tactics).
#
# Outputs (in reports/ by default):
#   - distribution_stance.csv
#   - distribution_tactics.csv
#   - distribution_stance_non_neutral.csv
#   - distribution_tactics_non_neutral.csv
#   - stats_stance_scores.csv
#   - stats_tactic_scores.csv
#   - pivot_stance_x_tactic_counts.csv
#   - pivot_stance_x_tactic_pct.csv
#   - top_stance_<label>.csv (top-N examples per non-neutral stance label)
#   - top_tactic_<label>.csv (top-N examples per non-neutral tactic label)
#   - summary.md (human-readable snapshot)
#
# Why this exists:
#   - You want fast visibility: what dominates, what’s rare but high-confidence,
#     and what combinations occur (stance × tactic).
#   - You want CSVs you can open instantly and a single summary.md you can paste
#     into README or attach to a report.
#
# Notes:
#   - The script is defensive: if columns are missing, it skips those outputs.
#   - It prefers text_en (translated) when present; otherwise falls back to text.
#   - "Top examples" are ranked by score and include a short preview for quick scanning.

import os
import argparse
import pandas as pd

DEFAULT_INPUT = "data/processed/master_dataset_with_sentiment_stance_tactics.csv"
OUT_DIR = "reports"

# Columns (change here if your names differ)
STANCE_LABEL_COL = "stance_label"
STANCE_SCORE_COL = "stance_score"
TACTIC_LABEL_COL = "tactic_label"
TACTIC_SCORE_COL = "tactic_score"
SENTIMENT_COL = "sentiment_label"

TEXT_COL_PRIMARY = "text_en"
TEXT_COL_FALLBACK = "text"


def pick_text_col(df: pd.DataFrame) -> str:
    """
    Choose which column to treat as the canonical text field.
    Preference:
      1) text_en if present and not empty
      2) text if present
      3) any column containing 'text' in the name
      4) "" (none found)
    """
    if TEXT_COL_PRIMARY in df.columns and df[TEXT_COL_PRIMARY].notna().any():
        return TEXT_COL_PRIMARY
    if TEXT_COL_FALLBACK in df.columns:
        return TEXT_COL_FALLBACK
    for c in df.columns:
        if "text" in c.lower():
            return c
    return ""


def pct_series(s: pd.Series) -> pd.DataFrame:
    """
    Value counts with percentage, standardized to:
      label | count | pct
    """
    vc = s.fillna("neutral").astype(str).value_counts(dropna=False)
    total = vc.sum() if vc.sum() else 1
    out = pd.DataFrame({
        "label": vc.index,
        "count": vc.values,
        "pct": (vc.values / total * 100.0).round(2),
    })
    return out


def stats_by_label(df: pd.DataFrame, label_col: str, score_col: str) -> pd.DataFrame:
    """
    Summary statistics of a score grouped by a categorical label:
      count, mean, median, min, max
    """
    if label_col not in df.columns or score_col not in df.columns:
        return pd.DataFrame()

    tmp = df[[label_col, score_col]].copy()
    tmp[label_col] = tmp[label_col].fillna("neutral").astype(str)
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce").fillna(0.0)

    g = tmp.groupby(label_col)[score_col].agg(["count", "mean", "median", "min", "max"]).reset_index()
    g = g.sort_values("count", ascending=False)

    # Pretty formatting for markdown/CSV readability
    g["mean"] = g["mean"].round(4)
    g["median"] = g["median"].round(4)
    g["min"] = g["min"].round(4)
    g["max"] = g["max"].round(4)
    return g


def top_examples(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    text_col: str,
    label: str,
    n: int = 25
) -> pd.DataFrame:
    """
    Extract top-N highest scoring examples for a given label.
    Output columns:
      <label_col> | <score_col> | text_preview
    """
    if label_col not in df.columns or score_col not in df.columns or text_col not in df.columns:
        return pd.DataFrame()

    tmp = df[[label_col, score_col, text_col]].copy()
    tmp[label_col] = tmp[label_col].fillna("neutral").astype(str)
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce").fillna(0.0)
    tmp[text_col] = tmp[text_col].fillna("").astype(str)

    sub = tmp[tmp[label_col] == label].sort_values(score_col, ascending=False).head(n)
    sub = sub.rename(columns={text_col: "text"})

    # Preview column: optimized for "scan quickly in a spreadsheet"
    sub["text_preview"] = sub["text"].str.replace("\n", " ", regex=False).str.slice(0, 220)

    return sub[[label_col, score_col, "text_preview"]]


def safe_write(df: pd.DataFrame, path: str):
    """
    Write CSV only if df is non-empty.
    Keeps the reports folder clean (no empty files).
    """
    if df is None or df.empty:
        return
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--outdir", default=OUT_DIR)
    ap.add_argument("--topn", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    text_col = pick_text_col(df)

    # -------------------------
    # Distributions
    # -------------------------
    stance_dist = pct_series(df[STANCE_LABEL_COL]) if STANCE_LABEL_COL in df.columns else pd.DataFrame()
    tactic_dist = pct_series(df[TACTIC_LABEL_COL]) if TACTIC_LABEL_COL in df.columns else pd.DataFrame()

    # -------------------------
    # Score stats
    # -------------------------
    stance_stats = stats_by_label(df, STANCE_LABEL_COL, STANCE_SCORE_COL)
    tactic_stats = stats_by_label(df, TACTIC_LABEL_COL, TACTIC_SCORE_COL)

    # -------------------------
    # Non-neutral subsets (often the most useful lens)
    # -------------------------
    if STANCE_LABEL_COL in df.columns:
        stance_non = df[df[STANCE_LABEL_COL].fillna("neutral").astype(str) != "neutral"]
        stance_non_dist = pct_series(stance_non[STANCE_LABEL_COL]) if len(stance_non) else pd.DataFrame()
    else:
        stance_non = pd.DataFrame()
        stance_non_dist = pd.DataFrame()

    if TACTIC_LABEL_COL in df.columns:
        tactic_non = df[df[TACTIC_LABEL_COL].fillna("neutral").astype(str) != "neutral"]
        tactic_non_dist = pct_series(tactic_non[TACTIC_LABEL_COL]) if len(tactic_non) else pd.DataFrame()
    else:
        tactic_non = pd.DataFrame()
        tactic_non_dist = pd.DataFrame()

    # -------------------------
    # Pivot: stance × tactic
    # -------------------------
    pivot = pd.DataFrame()
    if STANCE_LABEL_COL in df.columns and TACTIC_LABEL_COL in df.columns:
        pivot = pd.crosstab(
            df[STANCE_LABEL_COL].fillna("neutral").astype(str),
            df[TACTIC_LABEL_COL].fillna("neutral").astype(str),
            normalize=False
        ).reset_index().rename(columns={STANCE_LABEL_COL: "stance"})

        pivot_pct = pd.crosstab(
            df[STANCE_LABEL_COL].fillna("neutral").astype(str),
            df[TACTIC_LABEL_COL].fillna("neutral").astype(str),
            normalize=True
        ) * 100.0
        pivot_pct = pivot_pct.round(3).reset_index().rename(columns={STANCE_LABEL_COL: "stance"})
    else:
        pivot_pct = pd.DataFrame()

    # -------------------------
    # Top examples per label
    # -------------------------
    top_blocks = []
    if text_col:
        # Stance examples
        if STANCE_LABEL_COL in df.columns and STANCE_SCORE_COL in df.columns:
            for lbl in df[STANCE_LABEL_COL].fillna("neutral").astype(str).value_counts().index.tolist():
                if lbl == "neutral":
                    continue
                ex = top_examples(df, STANCE_LABEL_COL, STANCE_SCORE_COL, text_col, lbl, n=args.topn)
                if not ex.empty:
                    outp = os.path.join(args.outdir, f"top_stance_{lbl}.csv")
                    safe_write(ex, outp)
                    top_blocks.append(("stance", lbl, outp))

        # Tactic examples
        if TACTIC_LABEL_COL in df.columns and TACTIC_SCORE_COL in df.columns:
            for lbl in df[TACTIC_LABEL_COL].fillna("neutral").astype(str).value_counts().index.tolist():
                if lbl == "neutral":
                    continue
                ex = top_examples(df, TACTIC_LABEL_COL, TACTIC_SCORE_COL, text_col, lbl, n=args.topn)
                if not ex.empty:
                    outp = os.path.join(args.outdir, f"top_tactic_{lbl}.csv")
                    safe_write(ex, outp)
                    top_blocks.append(("tactic", lbl, outp))

    # -------------------------
    # Write CSV outputs
    # -------------------------
    safe_write(stance_dist, os.path.join(args.outdir, "distribution_stance.csv"))
    safe_write(tactic_dist, os.path.join(args.outdir, "distribution_tactics.csv"))
    safe_write(stance_non_dist, os.path.join(args.outdir, "distribution_stance_non_neutral.csv"))
    safe_write(tactic_non_dist, os.path.join(args.outdir, "distribution_tactics_non_neutral.csv"))
    safe_write(stance_stats, os.path.join(args.outdir, "stats_stance_scores.csv"))
    safe_write(tactic_stats, os.path.join(args.outdir, "stats_tactic_scores.csv"))
    safe_write(pivot, os.path.join(args.outdir, "pivot_stance_x_tactic_counts.csv"))
    safe_write(pivot_pct, os.path.join(args.outdir, "pivot_stance_x_tactic_pct.csv"))

    # -------------------------
    # Build summary.md
    # -------------------------
    n_rows = len(df)
    n_text = df[text_col].notna().sum() if text_col else 0

    stance_neutral_pct = (
        float(stance_dist.loc[stance_dist["label"] == "neutral", "pct"].values[0])
        if not stance_dist.empty and (stance_dist["label"] == "neutral").any()
        else None
    )
    tactic_neutral_pct = (
        float(tactic_dist.loc[tactic_dist["label"] == "neutral", "pct"].values[0])
        if not tactic_dist.empty and (tactic_dist["label"] == "neutral").any()
        else None
    )

    lines = []
    lines.append("# Results Snapshot\n")
    lines.append(f"- Rows: **{n_rows:,}**")
    if text_col:
        lines.append(f"- Text column: **{text_col}** (non-empty: {n_text:,})")
    if stance_neutral_pct is not None:
        lines.append(f"- Stance neutral share: **{stance_neutral_pct:.2f}%**")
    if tactic_neutral_pct is not None:
        lines.append(f"- Tactics neutral share: **{tactic_neutral_pct:.2f}%**")
    lines.append("")

    if not stance_dist.empty:
        lines.append("## Stance distribution\n")
        lines.append(stance_dist.to_markdown(index=False))
        lines.append("")
    if not tactic_dist.empty:
        lines.append("## Tactics distribution\n")
        lines.append(tactic_dist.to_markdown(index=False))
        lines.append("")

    if not stance_stats.empty:
        lines.append("## Stance score stats (by label)\n")
        lines.append(stance_stats.to_markdown(index=False))
        lines.append("")
    if not tactic_stats.empty:
        lines.append("## Tactic score stats (by label)\n")
        lines.append(tactic_stats.to_markdown(index=False))
        lines.append("")

    if not pivot.empty:
        lines.append("## Stance × Tactic (counts)\n")
        lines.append(pivot.to_markdown(index=False))
        lines.append("")

    if not pivot_pct.empty:
        lines.append("## Stance × Tactic (percent of total)\n")
        lines.append(pivot_pct.to_markdown(index=False))
        lines.append("")

    if top_blocks:
        lines.append("## Top examples (saved as CSV)\n")
        for kind, lbl, path in top_blocks:
            lines.append(f"- {kind}: **{lbl}** -> `{path}`")
        lines.append("")

    summary_path = os.path.join(args.outdir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote reports to: {args.outdir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
