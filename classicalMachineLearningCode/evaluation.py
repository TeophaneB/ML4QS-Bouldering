from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
REQUIRED_COLUMNS = {
    "classifier",
    "selector",
    "avg_performance_v",
    "outer_scores_per_repeat",
    "best_parameters",
}
METRICS = [
    "test_accuracy",
    "test_precision_macro",
    "test_recall_macro",
    "test_f1_macro",
]


def extract_window_size(file_path: Path) -> int | None:
    match = re.search(r"summary_(\d+)", file_path.name)
    if not match:
        return None
    return int(match.group(1))


def _extend_numeric(target: list[float], values) -> None:
    if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        iterable = values
    else:
        iterable = [values]

    for value in iterable:
        try:
            target.append(float(value))
        except (TypeError, ValueError):
            continue


def parse_outer_scores(raw_value, source_file: str, row_index: int) -> dict[str, list[float]] | None:
    try:
        parsed = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        print(f"Warning: skipped row {row_index} in {source_file} because outer_scores_per_repeat could not be parsed")
        return None

    if isinstance(parsed, dict):
        score_blocks = [parsed]
    elif isinstance(parsed, (list, tuple)):
        score_blocks = list(parsed)
    else:
        print(f"Warning: skipped row {row_index} in {source_file} because outer_scores_per_repeat has an unexpected type")
        return None

    metric_values: dict[str, list[float]] = {metric: [] for metric in METRICS}

    for block in score_blocks:
        if not isinstance(block, dict):
            continue
        for metric in METRICS:
            if metric in block:
                _extend_numeric(metric_values[metric], block[metric])

    if any(not values for values in metric_values.values()):
        print(f"Warning: skipped row {row_index} in {source_file} because one or more metrics were missing")
        return None

    return metric_values


def summarize_row(row: pd.Series, window_size: int, source_file: str, row_index: int) -> dict | None:
    metric_values = parse_outer_scores(row["outer_scores_per_repeat"], source_file, row_index)
    if metric_values is None:
        return None

    summary = {
        "window_size": window_size,
        "classifier": row["classifier"],
        "selector": row["selector"],
        "best_parameters": row["best_parameters"],
        "source_file": source_file,
    }

    for metric in METRICS:
        values = np.asarray(metric_values[metric], dtype=float)
        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=0))

        if metric == "test_accuracy":
            summary["mean_accuracy"] = mean_value
            summary["std_accuracy"] = std_value
        elif metric == "test_precision_macro":
            summary["mean_precision_macro"] = mean_value
            summary["std_precision_macro"] = std_value
        elif metric == "test_recall_macro":
            summary["mean_recall_macro"] = mean_value
            summary["std_recall_macro"] = std_value
        elif metric == "test_f1_macro":
            summary["mean_f1_macro"] = mean_value
            summary["std_f1_macro"] = std_value

    return summary


def load_all_results(results_dir: Path) -> pd.DataFrame:
    all_rows: list[dict] = []

    for file_path in sorted(results_dir.glob("*.tsv")):
        window_size = extract_window_size(file_path)
        if window_size is None:
            print(f"Warning: skipped {file_path.name} because the window size could not be extracted")
            continue

        try:
            df = pd.read_csv(file_path, sep="\t")
        except Exception as exc:  # pragma: no cover - defensive guard for malformed files
            print(f"Warning: skipped {file_path.name} because it could not be read ({exc})")
            continue

        missing_columns = REQUIRED_COLUMNS - set(df.columns)
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            print(f"Warning: skipped {file_path.name} because required columns are missing: {missing_text}")
            continue

        for row_index, row in df.iterrows():
            summary = summarize_row(row, window_size, file_path.name, row_index)
            if summary is not None:
                all_rows.append(summary)

    if not all_rows:
        return pd.DataFrame()

    all_results = pd.DataFrame(all_rows)
    all_results = all_results.sort_values(
        ["window_size", "selector", "classifier"],
        kind="stable",
    ).reset_index(drop=True)
    return all_results


def build_best_per_window(all_results: pd.DataFrame) -> pd.DataFrame:
    if all_results.empty:
        return pd.DataFrame()

    best_rows = []
    for window_size, window_df in all_results.groupby("window_size", sort=True):
        best_index = window_df["mean_f1_macro"].idxmax()
        best_row = window_df.loc[best_index, [
            "window_size",
            "selector",
            "classifier",
            "mean_f1_macro",
            "std_f1_macro",
            "mean_accuracy",
            "source_file",
            "best_parameters",
        ]]
        best_rows.append(best_row.to_dict())

    best_per_window = pd.DataFrame(best_rows).sort_values("window_size", kind="stable").reset_index(drop=True)
    return best_per_window


def save_bar_chart(best_per_window: pd.DataFrame, output_path: Path) -> None:
    if best_per_window.empty:
        print("Warning: no data available for the best-model bar chart")
        return

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(best_per_window)), 6))
    x_positions = np.arange(len(best_per_window))
    bars = ax.bar(x_positions, best_per_window["mean_f1_macro"], color="#5BCF18")

    ax.set_title("Best Macro F1 Score per Window Size")
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Mean Outer-CV Macro F1")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(best_per_window["window_size"].astype(str))

    for bar, (_, row) in zip(bars, best_per_window.iterrows()):
        label = f"{row['classifier']} + {row['selector']}\n{row['mean_f1_macro']:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )

    ax.set_ylim(0, float(best_per_window["mean_f1_macro"].max()) + 0.28)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(best_window_df: pd.DataFrame, window_size: int, output_path: Path) -> None:
    if best_window_df.empty:
        print("Warning: no data available for the heatmap")
        return

    selectors = sorted(best_window_df["selector"].dropna().unique().tolist())
    classifiers = sorted(best_window_df["classifier"].dropna().unique().tolist())

    pivot = best_window_df.pivot_table(
        index="selector",
        columns="classifier",
        values="mean_f1_macro",
        aggfunc="mean",
    ).reindex(index=selectors, columns=classifiers)

    best_index = best_window_df["mean_f1_macro"].idxmax()
    best_row = best_window_df.loc[best_index]

    matrix = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)

    fig_width = max(8, 1.3 * len(classifiers))
    fig_height = max(6, 0.7 * len(selectors) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#E6E6E6")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, label="Mean Outer-CV Macro F1")

    ax.set_xticks(np.arange(len(classifiers)))
    ax.set_yticks(np.arange(len(selectors)))
    ax.set_xticklabels(classifiers, rotation=45, ha="right")
    ax.set_yticklabels(selectors)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Feature selection / reduction method")
    ax.set_title(
        f"Mean Outer-CV Macro F1 by Selection Method and Classifier (Best Window Size = {window_size})"
    )

    for row_idx, selector in enumerate(selectors):
        for col_idx, classifier in enumerate(classifiers):
            value = pivot.loc[selector, classifier]
            if pd.isna(value):
                text = "NA"
            else:
                text = f"{value:.3f}"
                if selector == best_row["selector"] and classifier == best_row["classifier"]:
                    text += "*"

            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_selection_pairs_by_window_size(all_results: pd.DataFrame, output_path: Path) -> None:
    if all_results.empty:
        print("Warning: no data available for the selection-pair comparison plot")
        return

    # Use the column names already produced by this script, but keep a small fallback.
    if "selector" in all_results.columns:
        selection_col = "selector"
    elif "feature_selection" in all_results.columns:
        selection_col = "feature_selection"
    else:
        raise KeyError("No selector/feature_selection column found in results DataFrame")

    if "mean_f1_macro" in all_results.columns:
        metric_col = "mean_f1_macro"
        y_label = "Mean F1-score"
    elif "mean_f1" in all_results.columns:
        metric_col = "mean_f1"
        y_label = "Mean F1-score"
    elif "mean_score" in all_results.columns:
        metric_col = "mean_score"
        y_label = "Mean score"
    else:
        raise KeyError("No mean_f1_macro, mean_f1, or mean_score column found in results DataFrame")

    plot_df = all_results.copy()
    plot_df["model_selection_pair"] = plot_df["classifier"].astype(str) + " + " + plot_df[selection_col].astype(str)
    plot_df = plot_df.sort_values(["window_size", "model_selection_pair"], kind="stable")

    window_sizes = sorted(plot_df["window_size"].dropna().unique().tolist())
    pairs = plot_df["model_selection_pair"].dropna().unique().tolist()

    fig_width = max(12, 1.2 * len(window_sizes) + 0.6 * len(pairs))
    fig, (ax, ax_side) = plt.subplots(
        ncols=2,
        figsize=(fig_width, 6),
        gridspec_kw={"width_ratios": [4, 1.8]},
    )

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, max(1, len(pairs))))

    for idx, pair in enumerate(pairs):
        pair_df = plot_df[plot_df["model_selection_pair"] == pair].sort_values("window_size")
        x_values = pair_df["window_size"].to_numpy()
        y_values = pair_df[metric_col].to_numpy(dtype=float)

        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.8,
            color=colors[idx],
            label=pair,
        )

    ax.set_title("Classifier and Feature Selection Pair Performance by Window Size")
    ax.set_xlabel("Window size")
    ax.set_ylabel(y_label)
    ax.set_xscale("log")
    ax.set_xticks(window_sizes)
    ax.set_xticklabels([str(window_size) for window_size in window_sizes])
    ax.tick_params(axis="x", rotation=30)

    handles, labels = ax.get_legend_handles_labels()
    ax_side.axis("off")
    ax_side.legend(handles, labels, loc="upper left", title="Model + selection pair")

    mean_by_window = plot_df.groupby("window_size", sort=True)[metric_col].mean()
    best_window = int(mean_by_window.idxmax())
    if metric_col in {"mean_f1_macro", "mean_f1"}:
        summary_title = "Average macro-F1 by window"
    else:
        summary_title = "Average score by window"

    summary_lines = [summary_title]
    for window_size, avg_score in mean_by_window.items():
        window_int = int(window_size)
        best_suffix = "  <- best" if window_int == best_window else ""
        summary_lines.append(f"{window_int}: {avg_score:.3f}{best_suffix}")

    ax_side.text(
        0.02,
        0.02,
        "\n".join(summary_lines),
        transform=ax_side.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F7F7F7", "edgecolor": "#666666"},
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_best_model(best_row: pd.Series) -> None:
    print("\nBest overall model")
    print(f"best window size: {best_row['window_size']}")
    print(f"best selector: {best_row['selector']}")
    print(f"best classifier: {best_row['classifier']}")
    print(f"mean macro F1: {best_row['mean_f1_macro']:.3f}")
    print(f"standard deviation macro F1: {best_row['std_f1_macro']:.3f}")
    print(f"mean accuracy: {best_row['mean_accuracy']:.3f}")
    print(f"source file: {best_row['source_file']}")
    print(f"best parameters: {best_row['best_parameters']}")


def build_best_model_filename(best_row: pd.Series) -> str:
    window_size = str(best_row["window_size"])
    classifier = str(best_row["classifier"])
    selector = str(best_row["selector"])
    safe_classifier = classifier.replace("/", "-")
    safe_selector = selector.replace("/", "-")
    return f"best_model_found_window({window_size})_({safe_classifier}, {safe_selector}).csv"

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = load_all_results(RESULTS_DIR)
    if all_results.empty:
        print("Warning: no result rows were parsed")
        return

    best_per_window = build_best_per_window(all_results)

    all_results_path = RESULTS_DIR / "all_window_results_summary.csv"
    best_per_window_path = RESULTS_DIR / "best_model_per_window.csv"
    bar_chart_path = RESULTS_DIR / "best_macro_f1_per_window.png"
    heatmap_path = RESULTS_DIR / "best_window_macro_f1_heatmap.png"
    selection_pairs_path = RESULTS_DIR / "classification_selection_pairs_by_window_size.png"

    all_results.to_csv(all_results_path, index=False)
    best_per_window.to_csv(best_per_window_path, index=False)

    save_selection_pairs_by_window_size(all_results, selection_pairs_path)
    save_bar_chart(best_per_window, bar_chart_path)

    overall_best_index = all_results["mean_f1_macro"].idxmax()
    overall_best_row = all_results.loc[overall_best_index]
    print_best_model(overall_best_row)

    best_model_path = RESULTS_DIR / build_best_model_filename(overall_best_row)
    pd.DataFrame([overall_best_row]).to_csv(best_model_path, index=False)

    best_window_size = int(overall_best_row["window_size"])
    best_source_file = overall_best_row["source_file"]
    best_window_df = all_results[
        (all_results["window_size"] == best_window_size)
        & (all_results["source_file"] == best_source_file)
    ].copy()

    save_heatmap(best_window_df, best_window_size, heatmap_path)


if __name__ == "__main__":
    main()