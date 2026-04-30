"""Generate a unified one-page summary report across baselines and CNN-BiLSTM.

This script consolidates:
    - results/baselines/baseline_results.csv
    - results/cnn_bdlstm/cnn_bilstm_results.csv

into a single output folder containing:
    - unified_results.csv
    - one_pager_summary.md
    - one_pager_summary.html
    - plots/*.png + *.pdf

The output is designed for quick review after a single batch run that executes
both modeling scripts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load_results(baseline_csv: Path, cnn_csv: Path) -> pd.DataFrame:
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Missing baseline results: {baseline_csv}")
    if not cnn_csv.exists():
        raise FileNotFoundError(f"Missing CNN results: {cnn_csv}")

    baseline = pd.read_csv(baseline_csv)
    baseline["family"] = "baseline"
    baseline["model_display"] = baseline["model"]

    cnn = pd.read_csv(cnn_csv)
    cnn["family"] = "cnn_bdlstm"
    cnn["model_display"] = "CNN-Attn-BiLSTM"

    common_cols = sorted(set(baseline.columns).intersection(cnn.columns))
    combined = pd.concat(
        [baseline[common_cols], cnn[common_cols]],
        axis=0,
        ignore_index=True,
    )
    return combined


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def _plot_family_best_by_horizon(df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_style("whitegrid")

    # Best per family/horizon based on test AUPRC.
    idx = df.groupby(["family", "horizon"])["test_auprc"].idxmax()
    best = (
        df.loc[idx, ["family", "horizon", "channel_set", "model_display",
                     "test_auprc", "test_auroc", "test_balanced_accuracy",
                     "test_recall_sensitivity", "test_specificity", "test_f1"]]
        .sort_values(["horizon", "family"])
    )

    for metric, title in [
        ("test_auprc", "Best Test AUPRC by Horizon"),
        ("test_auroc", "Best Test AUROC by Horizon"),
        ("test_balanced_accuracy", "Best Test Balanced Accuracy by Horizon"),
        ("test_f1", "Best Test F1 by Horizon"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(
            data=best,
            x="horizon",
            y=metric,
            hue="family",
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Hours of data used")
        ax.set_ylabel(title.replace("Best ", ""))
        ax.set_title(title)
        _save_fig(fig, plots_dir, f"family_best_{metric}_by_horizon")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=best,
        x="test_specificity",
        y="test_recall_sensitivity",
        hue="family",
        style="horizon",
        s=120,
        ax=ax,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Best Family Models: Sensitivity vs Specificity by Horizon")
    _save_fig(fig, plots_dir, "family_best_sens_spec")


def _plot_channel_heatmaps(df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_style("white")
    for family in sorted(df["family"].unique()):
        fam = df[df["family"] == family]
        for metric, title in [
            ("test_auprc", "Test AUPRC"),
            ("test_auroc", "Test AUROC"),
            ("test_balanced_accuracy", "Test Balanced Accuracy"),
        ]:
            pivot = (
                fam.groupby(["channel_set", "horizon"])[metric]
                .max()
                .unstack("horizon")
                .sort_index()
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
            ax.set_title(f"{family}: best {title} per horizon/channel set")
            ax.set_xlabel("Horizon (hours)")
            ax.set_ylabel("Channel set")
            _save_fig(fig, plots_dir, f"{family}_{metric}_heatmap")


def _md_table(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind == "f":
            out[col] = out[col].map(lambda x: f"{x:.4f}")
    return out.to_markdown(index=False)


def _build_summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ranking_cols = [
        "family", "horizon", "channel_set", "model_display",
        "test_auprc", "test_auroc", "test_balanced_accuracy",
        "test_recall_sensitivity", "test_specificity", "test_f1", "test_brier",
        "val_threshold",
    ]
    top_overall = (
        df.sort_values(["test_auprc", "test_auroc"], ascending=False)
        .head(15)[ranking_cols]
        .reset_index(drop=True)
    )

    top_by_horizon = (
        df.loc[df.groupby("horizon")["test_auprc"].idxmax(), ranking_cols]
        .sort_values("horizon")
        .reset_index(drop=True)
    )

    best_family_horizon = (
        df.loc[df.groupby(["family", "horizon"])["test_auprc"].idxmax(), ranking_cols]
        .sort_values(["horizon", "family"])
        .reset_index(drop=True)
    )

    best_baseline_by_horizon = (
        df[df["family"] == "baseline"]
        .loc[df[df["family"] == "baseline"].groupby("horizon")["test_auprc"].idxmax(), ranking_cols]
        .sort_values("horizon")
        .reset_index(drop=True)
    )

    best_cnn_by_horizon = (
        df[df["family"] == "cnn_bdlstm"]
        .loc[df[df["family"] == "cnn_bdlstm"].groupby("horizon")["test_auprc"].idxmax(), ranking_cols]
        .sort_values("horizon")
        .reset_index(drop=True)
    )

    return {
        "top_overall": top_overall,
        "top_by_horizon": top_by_horizon,
        "best_family_horizon": best_family_horizon,
        "best_baseline_by_horizon": best_baseline_by_horizon,
        "best_cnn_by_horizon": best_cnn_by_horizon,
    }


def _write_markdown_report(
    out_path: Path,
    tables: dict[str, pd.DataFrame],
    plots_dir: Path,
) -> None:
    lines: list[str] = [
        "# Unified Sepsis Modeling Summary",
        "",
        "This report combines baseline-model and CNN-Attention-BiLSTM runs.",
        "",
        "## Top 15 Overall (ranked by test AUPRC)",
        "",
        _md_table(tables["top_overall"]),
        "",
        "## Best Model Per Horizon (overall)",
        "",
        _md_table(tables["top_by_horizon"]),
        "",
        "## Best Per Family and Horizon",
        "",
        _md_table(tables["best_family_horizon"]),
        "",
        "## Best Baseline Per Horizon",
        "",
        _md_table(tables["best_baseline_by_horizon"]),
        "",
        "## Best CNN Per Horizon",
        "",
        _md_table(tables["best_cnn_by_horizon"]),
        "",
        "## Plot Assets",
        "",
        f"- Combined plots: `{plots_dir}`",
        "",
        "Important generated figures:",
        "- `family_best_test_auprc_by_horizon.png`",
        "- `family_best_test_auroc_by_horizon.png`",
        "- `family_best_test_balanced_accuracy_by_horizon.png`",
        "- `family_best_sens_spec.png`",
        "- `<family>_test_auprc_heatmap.png`",
        "- `<family>_test_auroc_heatmap.png`",
        "- `<family>_test_balanced_accuracy_heatmap.png`",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_html_report(
    out_path: Path,
    tables: dict[str, pd.DataFrame],
    plots_dir: Path,
) -> None:
    def to_html_table(df: pd.DataFrame) -> str:
        fmt = df.copy()
        for c in fmt.columns:
            if fmt[c].dtype.kind == "f":
                fmt[c] = fmt[c].map(lambda x: f"{x:.4f}")
        return fmt.to_html(index=False, escape=False)

    imgs = [
        "family_best_test_auprc_by_horizon.png",
        "family_best_test_auroc_by_horizon.png",
        "family_best_test_balanced_accuracy_by_horizon.png",
        "family_best_test_f1_by_horizon.png",
        "family_best_sens_spec.png",
        "baseline_test_auprc_heatmap.png",
        "baseline_test_auroc_heatmap.png",
        "baseline_test_balanced_accuracy_heatmap.png",
        "cnn_bdlstm_test_auprc_heatmap.png",
        "cnn_bdlstm_test_auroc_heatmap.png",
        "cnn_bdlstm_test_balanced_accuracy_heatmap.png",
    ]

    css = """
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; color: #111; }
      h1, h2 { margin-top: 28px; }
      table { border-collapse: collapse; width: 100%; margin: 12px 0 24px 0; }
      th, td { border: 1px solid #ddd; padding: 8px; font-size: 12px; }
      th { background: #f4f4f8; text-align: left; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
      img { width: 100%; border: 1px solid #ddd; }
      .note { color: #444; margin-bottom: 12px; }
    </style>
    """

    sections = [
        ("Top 15 Overall (test AUPRC)", tables["top_overall"]),
        ("Best Model Per Horizon (overall)", tables["top_by_horizon"]),
        ("Best Per Family and Horizon", tables["best_family_horizon"]),
        ("Best Baseline Per Horizon", tables["best_baseline_by_horizon"]),
        ("Best CNN Per Horizon", tables["best_cnn_by_horizon"]),
    ]

    html = [f"<html><head>{css}</head><body>"]
    html.append("<h1>Unified Sepsis Modeling Summary</h1>")
    html.append("<p class='note'>Combined summary of baseline and CNN-BiLSTM experiments.</p>")

    for title, table in sections:
        html.append(f"<h2>{title}</h2>")
        html.append(to_html_table(table))

    html.append("<h2>Unified Visualizations</h2>")
    html.append("<div class='grid'>")
    for img in imgs:
        img_path = plots_dir / img
        if img_path.exists():
            html.append(f"<div><img src='plots/{img}' alt='{img}'><p>{img}</p></div>")
    html.append("</div>")
    html.append("</body></html>")

    out_path.write_text("\n".join(html), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("results/baselines/baseline_results.csv"),
    )
    parser.add_argument(
        "--cnn-csv",
        type=Path,
        default=Path("results/cnn_bdlstm/cnn_bilstm_results.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/unified_summary"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    combined = _load_results(args.baseline_csv, args.cnn_csv)
    combined.to_csv(args.out_dir / "unified_results.csv", index=False)

    _plot_family_best_by_horizon(combined, plots_dir)
    _plot_channel_heatmaps(combined, plots_dir)

    tables = _build_summary_tables(combined)
    _write_markdown_report(args.out_dir / "one_pager_summary.md", tables, plots_dir)
    _write_html_report(args.out_dir / "one_pager_summary.html", tables, plots_dir)

    print("Unified summary written to:", args.out_dir)
    print(" -", args.out_dir / "unified_results.csv")
    print(" -", args.out_dir / "one_pager_summary.md")
    print(" -", args.out_dir / "one_pager_summary.html")
    print(" -", plots_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

