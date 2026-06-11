from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ANALYSES_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "analyses"


def _window_sort_key(value):
    text = str(value)
    if text.lower() == "full":
        return (1, text)

    try:
        return (0, int(text))
    except (TypeError, ValueError):
        return (0, text)


def _normalize_model_name(model: str | None) -> str:
    if model == "w2v+kmeans":
        return "W2V"
    if model is None:
        return "Unknown"
    return str(model).upper()


def _load_comparison_type(analysis_dir: Path) -> str:
    config_file = analysis_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    comparison_type = config.get("comparison_type")
    if comparison_type is None:
        raise ValueError(f"Missing 'comparison_type' in {config_file}")

    return comparison_type


def _load_corpus_signature(analysis_dir: Path) -> tuple[int | None, int | None]:
    config_file = analysis_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    corpus = config.get("corpus", {})
    return corpus.get("seed"), corpus.get("number_of_documents")


def _find_analysis_dir(
    analyses_root: Path,
    seed: int,
    number_of_documents: int,
    comparison_type: str,
) -> Path:
    candidates: list[tuple[int, Path]] = []
    available_types: list[str] = []

    for config_file in analyses_root.glob("*/config.json"):
        try:
            with open(config_file, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except Exception:
            continue

        corpus = config.get("corpus", {})
        if corpus.get("seed") != seed:
            continue
        if corpus.get("number_of_documents") != number_of_documents:
            continue

        available_type = config.get("comparison_type")
        if available_type is not None:
            available_types.append(str(available_type))

        if config.get("comparison_type") != comparison_type:
            continue

        try:
            analysis_index = int(config_file.parent.name)
        except ValueError:
            analysis_index = -1

        candidates.append((analysis_index, config_file.parent))

    if not candidates:
        unique_available_types = sorted(set(available_types))
        available_types_text = (
            f" Available comparison types for this seed/docs: {unique_available_types}."
            if unique_available_types
            else ""
        )
        raise FileNotFoundError(
            f"No analysis found for seed={seed}, docs={number_of_documents}, comparison_type={comparison_type} in {analyses_root}.{available_types_text}"
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _window_to_token_count(window) -> str | int | None:
    text = str(window)
    if text.lower() == "full":
        return None

    if text.startswith("(") and text.endswith(")"):
        try:
            left, right = ast.literal_eval(text)
            return int(left) + int(right) + 1
        except Exception:
            return None

    try:
        return 2 * int(text) + 1
    except (TypeError, ValueError):
        return None


def _window_token_label(window: str) -> str:
    text = str(window)
    if text.lower() == "full":
        return "Full"

    token_count = _window_to_token_count(text)
    if token_count is None:
        return text

    return f"{int(token_count)}t"


def _collect_rows_from_analysis_dir(
    analysis_dir: Path,
    metric: str,
    expected_comparison_type: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    config_file = analysis_dir / "config.json"
    results_file = analysis_dir / "results.parquet"

    if not config_file.exists() or not results_file.exists():
        raise FileNotFoundError(
            f"Expected config.json and results.parquet in {analysis_dir}"
        )

    comparison_type = _load_comparison_type(analysis_dir)
    if comparison_type != expected_comparison_type:
        raise ValueError(
            f"{analysis_dir.name} has comparison_type={comparison_type}, expected {expected_comparison_type}"
        )

    df = pd.read_parquet(results_file)
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    required_cols = {"model_x", "model_y", "window_x", "window_y"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    if comparison_type == "w2v_vs_w2v":
        # For W2V self-comparison, we only want the same W2V window.
        df = df[df["window_x"].astype(str) == df["window_y"].astype(str)].copy()

        df = df[
            (df["model_x"] == "w2v+kmeans")
            & (df["model_y"] == "w2v+kmeans")
        ].copy()

        if df.empty:
            return pd.DataFrame()

        for _, row in df.iterrows():
            rows.append(
                {
                    "analysis": analysis_dir.name,
                    "series": "W2V × W2V",
                    "w2v_window": str(row["window_x"]),
                    "sbm_window": None,
                    "metric": float(row[metric]),
                }
            )

    elif comparison_type == "sbm_vs_w2v":
        # For SBM × W2V, do NOT filter window_x == window_y.
        # We need all SBM windows for each W2V window to find the best SBM match.
        df = df[
            (df["model_x"] == "sbm")
            & (df["model_y"] == "w2v+kmeans")
        ].copy()

        if df.empty:
            return pd.DataFrame()

        for _, row in df.iterrows():
            rows.append(
                {
                    "analysis": analysis_dir.name,
                    "series": "W2V × SBM",
                    "w2v_window": str(row["window_y"]),
                    "sbm_window": str(row["window_x"]),
                    "metric": float(row[metric]),
                }
            )

    else:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)

def _summarize_w2v_self(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["w2v_window", "w2v_self_metric"])

    summary = (
        df.groupby("w2v_window", dropna=False)["metric"]
        .mean()
        .reset_index()
        .rename(columns={"metric": "w2v_self_metric"})
    )
    return summary


def _summarize_sbm_cross(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["w2v_window", "sbm_window", "sbm_metric"])

    rows: list[dict[str, object]] = []

    for w2v_window, group in df.groupby("w2v_window", dropna=False):
        if group.empty:
            continue

        best_metric = group["metric"].max()
        best_rows = group[group["metric"] == best_metric].copy()

        # Stable tie-break: prefer smaller token window if multiple SBM windows have same score.
        best_rows["sbm_token_count"] = best_rows["sbm_window"].map(_window_to_token_count)

        best_row = best_rows.sort_values(
            ["sbm_token_count", "sbm_window"],
            na_position="last",
        ).iloc[0]

        rows.append(
            {
                "w2v_window": str(w2v_window),
                "sbm_window": str(best_row["sbm_window"]),
                "sbm_metric": float(best_row["metric"]),
            }
        )

    return pd.DataFrame(rows)

def plot_dual_model_window_metric(
    seed: int,
    number_of_samples: int,
    metric: str = "nmi",
):
    w2v_analysis_dir = _find_analysis_dir(
        DEFAULT_ANALYSES_ROOT,
        seed,
        number_of_samples,
        comparison_type="w2v_vs_w2v",
    )
    sbm_analysis_dir = _find_analysis_dir(
        DEFAULT_ANALYSES_ROOT,
        seed,
        number_of_samples,
        comparison_type="sbm_vs_w2v",
    )

    print(
        f"[INFO] Seed/samples matched: seed={seed}, samples={number_of_samples}"
    )
    print(f"[INFO] Selected W2V analysis: {w2v_analysis_dir.name}")
    print(f"[INFO] Selected SBM analysis: {sbm_analysis_dir.name}")

    df_w2v_raw = _collect_rows_from_analysis_dir(
        w2v_analysis_dir,
        metric,
        expected_comparison_type="w2v_vs_w2v",
    )
    df_sbm_raw = _collect_rows_from_analysis_dir(
        sbm_analysis_dir,
        metric,
        expected_comparison_type="sbm_vs_w2v",
    )

    df_w2v = _summarize_w2v_self(df_w2v_raw)
    df_sbm = _summarize_sbm_cross(df_sbm_raw)

    if df_w2v.empty and df_sbm.empty:
        print(
            f"[WARN] No data found for metric '{metric}' in {w2v_analysis_dir} and {sbm_analysis_dir}"
        )
        return False

    windows = sorted(
        set(df_w2v["w2v_window"].dropna().astype(str).unique())
        | set(df_sbm["w2v_window"].dropna().astype(str).unique()),
        key=_window_sort_key,
    )
    x_map = {window: idx for idx, window in enumerate(windows)}


    fig, (ax_metric, ax_sbm) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3.2, 1.4],
            "hspace": 0.08,
        },
    )

    # For each W2V window, show the self-comparison and the best SBM cross-comparison.
    top_w2v = df_w2v.set_index("w2v_window").reindex(windows)
    top_sbm = df_sbm.set_index("w2v_window").reindex(windows)

    w2v_self_values = pd.to_numeric(
        top_w2v["w2v_self_metric"],
        errors="coerce",
    )
    sbm_best_values = pd.to_numeric(
        top_sbm["sbm_metric"],
        errors="coerce",
    )

    x_values = list(range(len(windows)))

    # -------------------------
    # Top plot: metric values
    # -------------------------
    ax_metric.scatter(
        x_values,
        w2v_self_values,
        color="#1f77b4",
        marker="o",
        s=70,
        label="W2V",
        zorder=3,
    )

    ax_metric.scatter(
        x_values,
        sbm_best_values,
        color="#d62728",
        marker="s",
        s=70,
        label="SBM",
        zorder=4,
    )

    # Annotate W2V self-comparison values
    for window in windows:
        if window not in top_w2v.index:
            continue

        if pd.isna(top_w2v.loc[window, "w2v_self_metric"]):
            continue

        x_value = x_map[window]
        y_value = float(top_w2v.loc[window, "w2v_self_metric"])

        ax_metric.annotate(
            f"{y_value:.3f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(-8, -8),
            ha="right",
            va="top",
            fontsize=8,
            color="#1f77b4",
        )

    for window in windows:
        if window not in top_sbm.index:
            continue

        if pd.isna(top_sbm.loc[window, "sbm_metric"]):
            continue

        x_value = x_map[window]
        y_value = top_sbm.loc[window, "sbm_metric"]
        sbm_window = top_sbm.loc[window, "sbm_window"]

        ax_metric.annotate(
            f"{float(y_value):.3f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(8, -5),
            ha="left",
            va="top",
            fontsize=8,
            color="#d62728",
        )

    metric_values = [
        float(value)
        for value in list(w2v_self_values.dropna()) + list(sbm_best_values.dropna())
    ]

    if metric_values:
        y_min = max(0.0, min(metric_values) - 0.05)
        y_max = min(1.0, max(metric_values) + 0.05)

        # Avoid an overly compressed top plot when values are very close.
        if y_max - y_min < 0.08:
            center = (y_min + y_max) / 2
            y_min = max(0.0, center - 0.04)
            y_max = min(1.0, center + 0.04)

        ax_metric.set_ylim(y_min, y_max)

    ax_metric.set_ylabel(metric.upper())
    ax_metric.set_title(
        f"W2V window on X, {metric.upper()} on Y: W2V self vs best SBM match"
    )
    ax_metric.grid(True, axis="y", alpha=0.25)

    # Hide x labels on the top plot because they will be shown on the middle line.
    ax_metric.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_metric.spines["bottom"].set_visible(False)

    # -------------------------
    # Bottom plot: best SBM window
    # -------------------------
    sbm_df = top_sbm.reset_index().rename(columns={"index": "w2v_window"})
    sbm_df = sbm_df[sbm_df["sbm_window"].notna()].copy()

    sbm_points = []

    for _, row in sbm_df.iterrows():
        w2v_window = str(row["w2v_window"])
        sbm_window = str(row["sbm_window"])

        if w2v_window not in x_map:
            continue

        token_count = _window_to_token_count(sbm_window)

        if token_count is None:
            continue

        sbm_points.append(
            {
                "x": x_map[w2v_window],
                "token_count": int(token_count),
            }
        )

    if sbm_points:
        unique_token_counts = sorted(
            {point["token_count"] for point in sbm_points}
        )

        # Fixed categorical spacing for the bottom plot.
        y_map = {
            token_count: idx
            for idx, token_count in enumerate(unique_token_counts)
        }

        sbm_x_values = [point["x"] for point in sbm_points]
        sbm_y_values = [y_map[point["token_count"]] for point in sbm_points]

        ax_sbm.scatter(
            sbm_x_values,
            sbm_y_values,
            color="#d62728",
            marker="s",
            s=70,
            label="_nolegend_",
            zorder=5,
        )

        for point in sbm_points:
            ax_sbm.annotate(
                f"{point['token_count']}t",
                (point["x"], y_map[point["token_count"]]),
                textcoords="offset points",
                xytext=(0, -12),
                ha="center",
                fontsize=8,
                color="#d62728",
            )

        ax_sbm.set_yticks(range(len(unique_token_counts)))
        ax_sbm.set_yticklabels(
            [f"{token_count}t" for token_count in unique_token_counts]
        )

        # Keeps smaller windows closer to the middle line and larger windows lower.
        ax_sbm.invert_yaxis()
        ax_sbm.set_ylim(len(unique_token_counts) - 0.5, -0.5)

    else:
        ax_sbm.text(
            0.5,
            0.5,
            "No W2V × SBM rows found.",
            transform=ax_sbm.transAxes,
            ha="center",
            va="center",
        )

    ax_sbm.set_ylabel("Best SBM window")
    ax_sbm.grid(True, axis="y", alpha=0.25)

    # Put W2V token labels on the middle line between both plots.
    x_labels = [_window_token_label(window) for window in windows]
    ax_sbm.set_xticks(x_values)
    ax_sbm.set_xticklabels(x_labels)

    ax_sbm.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        pad=4,
    )

    ax_sbm.xaxis.set_label_position("top")
    ax_sbm.set_xlabel("W2V window", labelpad=8)

    # Make the middle separation line clearer.
    ax_sbm.spines["top"].set_visible(True)
    ax_sbm.spines["top"].set_linewidth(1.2)

    handles_metric, labels_metric = ax_metric.get_legend_handles_labels()

    ax_metric.legend(
        handles_metric,
        labels_metric,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        frameon=False,
    )

    metric_folder = metric.lower()
    out_dir = sbm_analysis_dir / "w2vxsbm"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"dual_model_window_{metric_folder}.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close(fig)

    print(f"[INFO] Saved: {out_file}")
    print(f"[INFO] W2V summary rows: {len(df_w2v)}")
    print(f"[INFO] SBM summary rows: {len(df_sbm)}")
    print(f"[INFO] W2V windows: {windows}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Corpus seed used to locate the analyses.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        required=True,
        help="Number of samples used to locate the analyses.",
    )
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        default="nmi",
        help="Metric to plot (default: nmi).",
    )

    args = parser.parse_args()

    plot_dual_model_window_metric(
        args.seed,
        args.samples,
        args.metric,
    )


if __name__ == "__main__":
    main()