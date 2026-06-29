from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ANALYSES_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "analyses"

MODEL_LABELS = {
    "w2v+kmeans": "W2V",
    "sbm": "SBM",
    "random_k": "Random",
}

MODEL_COLORS = {
    "w2v+kmeans": "#1f77b4",
    "sbm": "#d62728",
    "random_k": "#2ca02c",
}

MODEL_MARKERS = {
    "w2v+kmeans": "o",
    "sbm": "s",
    "random_k": "^",
}


def _window_sort_key(value):
    text = str(value)
    if text.lower() == "full":
        return (1, text)

    try:
        return (0, int(text))
    except (TypeError, ValueError):
        return (0, text)


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

def _load_is_context(analysis_dir: Path | None) -> bool:
    if analysis_dir is None:
        return False

    config_file = analysis_dir / "config.json"
    if not config_file.exists():
        return False

    with open(config_file, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    return bool(config.get("graph", {}).get("context", False))


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
            f"No analysis found for seed={seed}, docs={number_of_documents}, "
            f"comparison_type={comparison_type} in {analyses_root}.{available_types_text}"
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]



def _try_find_analysis_dir(
    analyses_root: Path,
    seed: int,
    number_of_documents: int,
    comparison_type: str,
) -> Path | None:
    try:
        return _find_analysis_dir(
            analyses_root,
            seed,
            number_of_documents,
            comparison_type,
        )
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}")
        return None

def _window_to_token_count(window, is_context: bool = False) -> int | None:
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
        value = int(text)
    except (TypeError, ValueError):
        return None

    if is_context:
        return value

    return 2 * value + 1


def _window_token_label(window: str, is_context: bool = False) -> str:
    text = str(window)
    if text.lower() == "full":
        return "Full"

    token_count = _window_to_token_count(text, is_context=is_context)
    if token_count is None:
        return text

    return f"{int(token_count)}t"


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, str(model).upper())


def _dimension_label(is_context: bool) -> str:
    return "context size" if is_context else "window"


def _load_metric_dataframe(
    analysis_dir: Path,
    metric: str,
    expected_comparison_type: str,
) -> pd.DataFrame:
    config_file = analysis_dir / "config.json"
    results_file = analysis_dir / "results.parquet"

    if not config_file.exists() or not results_file.exists():
        raise FileNotFoundError(
            f"Expected config.json and results.parquet in {analysis_dir}"
        )

    comparison_type = _load_comparison_type(analysis_dir)
    if comparison_type != expected_comparison_type:
        raise ValueError(
            f"{analysis_dir.name} has comparison_type={comparison_type}, "
            f"expected {expected_comparison_type}"
        )

    df = pd.read_parquet(results_file)
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    required_cols = {"model_x", "model_y", "window_x", "window_y"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    return df.copy()


def _collect_self_rows(
    analysis_dir: Path | None,
    metric: str,
    expected_comparison_type: str,
    model_name: str,
) -> pd.DataFrame:
    if analysis_dir is None:
        return pd.DataFrame(columns=["anchor_window", "metric"])

    df = _load_metric_dataframe(analysis_dir, metric, expected_comparison_type)
    if df.empty:
        return pd.DataFrame(columns=["anchor_window", "metric"])

    df = df[
        (df["model_x"] == model_name)
        & (df["model_y"] == model_name)
        & (df["window_x"].astype(str) == df["window_y"].astype(str))
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=["anchor_window", "metric"])

    return pd.DataFrame(
        {
            "anchor_window": df["window_x"].astype(str),
            "metric": pd.to_numeric(df[metric], errors="coerce"),
        }
    ).dropna(subset=["metric"])


def _collect_cross_rows(
    analysis_dir: Path | None,
    metric: str,
    expected_comparison_type: str,
    anchor_model: str,
    candidate_model: str,
) -> pd.DataFrame:
    """
    Collect rows as anchor window x candidate window, independent of whether the
    anchor model is stored in model_x or model_y.
    """
    if analysis_dir is None:
        return pd.DataFrame(columns=["anchor_window", "candidate_window", "metric"])

    df = _load_metric_dataframe(analysis_dir, metric, expected_comparison_type)
    if df.empty:
        return pd.DataFrame(columns=["anchor_window", "candidate_window", "metric"])

    rows: list[dict[str, object]] = []

    direct = df[(df["model_x"] == anchor_model) & (df["model_y"] == candidate_model)]
    for _, row in direct.iterrows():
        rows.append(
            {
                "anchor_window": str(row["window_x"]),
                "candidate_window": str(row["window_y"]),
                "metric": float(row[metric]),
            }
        )

    reverse = df[(df["model_x"] == candidate_model) & (df["model_y"] == anchor_model)]
    for _, row in reverse.iterrows():
        rows.append(
            {
                "anchor_window": str(row["window_y"]),
                "candidate_window": str(row["window_x"]),
                "metric": float(row[metric]),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["anchor_window", "candidate_window", "metric"])

    return pd.DataFrame(rows)


def _summarize_self(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["anchor_window", column_name])

    return (
        df.groupby("anchor_window", dropna=False)["metric"]
        .mean()
        .reset_index()
        .rename(columns={"metric": column_name})
    )


def _summarize_best_cross(
    df: pd.DataFrame,
    *,
    candidate_window_column: str,
    metric_column: str,
    is_context: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["anchor_window", candidate_window_column, metric_column]
        )

    clean_df = df.copy()
    clean_df["anchor_window"] = clean_df["anchor_window"].astype(str)
    clean_df["candidate_window"] = clean_df["candidate_window"].astype(str)
    clean_df["metric"] = pd.to_numeric(clean_df["metric"], errors="coerce")
    clean_df = clean_df.dropna(subset=["metric"])

    if clean_df.empty:
        return pd.DataFrame(
            columns=["anchor_window", candidate_window_column, metric_column]
        )

    # The heatmaps show the mean value for each window pair. Therefore the dual
    # plot must first reproduce the same aggregation and only then select the
    # best candidate window for each anchor window.
    pair_means = (
        clean_df.groupby(["anchor_window", "candidate_window"], dropna=False)["metric"]
        .mean()
        .reset_index()
    )

    rows: list[dict[str, object]] = []

    for anchor_window, group in pair_means.groupby("anchor_window", dropna=False):
        if group.empty:
            continue

        best_metric = group["metric"].max()
        best_rows = group[group["metric"] == best_metric].copy()
        best_rows["candidate_token_count"] = best_rows["candidate_window"].map(
            lambda w: _window_to_token_count(w, is_context=is_context)
        )

        # Stable tie-break: prefer smaller candidate token window, keeping Full last.
        best_row = best_rows.sort_values(
            ["candidate_token_count", "candidate_window"],
            na_position="last",
        ).iloc[0]

        rows.append(
            {
                "anchor_window": str(anchor_window),
                candidate_window_column: str(best_row["candidate_window"]),
                metric_column: float(best_row["metric"]),
            }
        )

    return pd.DataFrame(rows)

def _metric_axis_limits(series_list: list[pd.Series]) -> tuple[float, float] | None:
    values: list[float] = []
    for series in series_list:
        if series is None:
            continue
        values.extend([float(value) for value in series.dropna().tolist()])

    if not values:
        return None

    y_min = max(0.0, min(values) - 0.05)
    y_max = min(1.0, max(values) + 0.05)

    if y_max - y_min < 0.08:
        center = (y_min + y_max) / 2
        y_min = max(0.0, center - 0.04)
        y_max = min(1.0, center + 0.04)

    return y_min, y_max



def _plot_anchor_candidate(
    *,
    seed: int,
    number_of_samples: int,
    metric: str,
    anchor_model: str,
    candidate_model: str,
    df_anchor_self: pd.DataFrame,
    df_candidate_best: pd.DataFrame,
    candidate_window_column: str,
    candidate_metric_column: str,
    output_dir: Path,
    folder_name: str,
    is_context: bool = False,
) -> bool:
    anchor_label = _model_label(anchor_model)
    candidate_label = _model_label(candidate_model)
    dimension_label = _dimension_label(is_context)

    windows = sorted(
        set(
            df_anchor_self.get("anchor_window", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
        )
        | set(
            df_candidate_best.get("anchor_window", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
        ),
        key=_window_sort_key,
    )

    if not windows:
        return False

    x_map = {window: idx for idx, window in enumerate(windows)}
    x_values = list(range(len(windows)))

    top_anchor = df_anchor_self.set_index("anchor_window").reindex(windows)
    top_candidate = df_candidate_best.set_index("anchor_window").reindex(windows)

    anchor_values = pd.to_numeric(
        top_anchor.get(
            f"{anchor_model}_self_metric",
            pd.Series(index=windows, dtype=float),
        ),
        errors="coerce",
    )
    candidate_values = pd.to_numeric(
        top_candidate.get(
            candidate_metric_column,
            pd.Series(index=windows, dtype=float),
        ),
        errors="coerce",
    )

    fig, (ax_metric, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.4], "hspace": 0.08},
    )

    ax_metric.scatter(
        x_values,
        anchor_values,
        color=MODEL_COLORS.get(anchor_model),
        marker=MODEL_MARKERS.get(anchor_model, "o"),
        s=70,
        label=anchor_label,
        zorder=3,
    )
    ax_metric.scatter(
        x_values,
        candidate_values,
        color=MODEL_COLORS.get(candidate_model),
        marker=MODEL_MARKERS.get(candidate_model, "s"),
        s=70,
        label=candidate_label,
        zorder=4,
    )

    for window in windows:
        x_value = x_map[window]

        if (
            window in top_anchor.index
            and pd.notna(top_anchor.loc[window].get(f"{anchor_model}_self_metric"))
        ):
            y_value = float(top_anchor.loc[window, f"{anchor_model}_self_metric"])
            ax_metric.annotate(
                f"{y_value:.3f}",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(-8, -8),
                ha="right",
                va="top",
                fontsize=8,
                color=MODEL_COLORS.get(anchor_model),
            )

        if (
            window in top_candidate.index
            and pd.notna(top_candidate.loc[window].get(candidate_metric_column))
        ):
            y_value = float(top_candidate.loc[window, candidate_metric_column])
            ax_metric.annotate(
                f"{y_value:.3f}",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(8, -5),
                ha="left",
                va="top",
                fontsize=8,
                color=MODEL_COLORS.get(candidate_model),
            )

    limits = _metric_axis_limits([anchor_values, candidate_values])
    if limits is not None:
        ax_metric.set_ylim(*limits)

    ax_metric.set_ylabel(metric.upper())
    ax_metric.set_title(
        f"{metric.upper()} | Seed:{seed} | Samples:{number_of_samples} | "
        f"{anchor_label} self-comparison vs {candidate_label} cross-comparison"
    )
    ax_metric.grid(True, axis="y", alpha=0.25)
    ax_metric.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_metric.spines["bottom"].set_visible(False)

    candidate_points = []
    candidate_df = top_candidate.reset_index().rename(columns={"index": "anchor_window"})
    candidate_df = candidate_df[candidate_df[candidate_window_column].notna()].copy()

    for _, row in candidate_df.iterrows():
        anchor_window = str(row["anchor_window"])
        candidate_window = str(row[candidate_window_column])
        if anchor_window not in x_map:
            continue

        candidate_points.append(
            {
                "x": x_map[anchor_window],
                "candidate_window": candidate_window,
                "candidate_label": _window_token_label(
                    candidate_window,
                    is_context=is_context,
                ),
            }
        )

    if candidate_points:
        unique_candidate_windows = sorted(
            {point["candidate_window"] for point in candidate_points},
            key=_window_sort_key,
        )
        y_map = {
            candidate_window: idx
            for idx, candidate_window in enumerate(unique_candidate_windows)
        }

        ax_bottom.scatter(
            [point["x"] for point in candidate_points],
            [y_map[point["candidate_window"]] for point in candidate_points],
            color=MODEL_COLORS.get(candidate_model),
            marker=MODEL_MARKERS.get(candidate_model, "s"),
            s=70,
            label="_nolegend_",
            zorder=5,
        )

        for point in candidate_points:
            ax_bottom.annotate(
                point["candidate_label"],
                (point["x"], y_map[point["candidate_window"]]),
                textcoords="offset points",
                xytext=(8, 0),
                ha="left",
                va="center",
                fontsize=8,
                color=MODEL_COLORS.get(candidate_model),
            )

        ax_bottom.set_yticks(range(len(unique_candidate_windows)))
        ax_bottom.set_yticklabels(
            [
                _window_token_label(window, is_context=is_context)
                for window in unique_candidate_windows
            ]
        )
        ax_bottom.set_ylim(len(unique_candidate_windows) - 0.5, -0.5)
    else:
        ax_bottom.text(
            0.5,
            0.5,
            f"No {anchor_label} × {candidate_label} rows found.",
            transform=ax_bottom.transAxes,
            ha="center",
            va="center",
        )

    ax_bottom.set_ylabel(f"Best {candidate_label} {dimension_label}")
    ax_bottom.grid(True, axis="y", alpha=0.25)

    x_labels = [_window_token_label(window, is_context=is_context) for window in windows]
    ax_bottom.set_xticks(x_values)
    ax_bottom.set_xticklabels(x_labels)
    ax_bottom.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        pad=4,
    )
    ax_bottom.xaxis.set_label_position("top")
    ax_bottom.set_xlabel(f"{anchor_label} {dimension_label}", labelpad=8)
    ax_bottom.spines["top"].set_visible(True)
    ax_bottom.spines["top"].set_linewidth(1.2)

    handles_metric, labels_metric = ax_metric.get_legend_handles_labels()
    ax_metric.legend(
        handles_metric,
        labels_metric,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        frameon=False,
    )

    out_dir = output_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"dual_model_window_{metric.lower()}.png"

    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.98, hspace=0.08)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved: {out_file}")
    return True


def _plot_w2v_sbm_random(
    *,
    seed: int,
    number_of_samples: int,
    metric: str,
    df_w2v_self: pd.DataFrame,
    df_sbm_best: pd.DataFrame,
    df_random_best: pd.DataFrame,
    output_dir: Path,
    is_context: bool = False,
) -> bool:
    dimension_label = _dimension_label(is_context)

    windows = sorted(
        set(
            df_w2v_self.get("anchor_window", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
        )
        | set(
            df_sbm_best.get("anchor_window", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
        )
        | set(
            df_random_best.get("anchor_window", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
        ),
        key=_window_sort_key,
    )

    if not windows:
        return False

    x_map = {window: idx for idx, window in enumerate(windows)}
    x_values = list(range(len(windows)))

    top_w2v = df_w2v_self.set_index("anchor_window").reindex(windows)
    top_sbm = df_sbm_best.set_index("anchor_window").reindex(windows)
    top_random = df_random_best.set_index("anchor_window").reindex(windows)

    w2v_values = pd.to_numeric(
        top_w2v.get(
            "w2v+kmeans_self_metric",
            pd.Series(index=windows, dtype=float),
        ),
        errors="coerce",
    )
    sbm_values = pd.to_numeric(
        top_sbm.get("sbm_metric", pd.Series(index=windows, dtype=float)),
        errors="coerce",
    )
    random_values = pd.to_numeric(
        top_random.get("random_metric", pd.Series(index=windows, dtype=float)),
        errors="coerce",
    )

    fig, (ax_metric, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.4], "hspace": 0.08},
    )

    ax_metric.scatter(
        x_values,
        w2v_values,
        color=MODEL_COLORS["w2v+kmeans"],
        marker=MODEL_MARKERS["w2v+kmeans"],
        s=70,
        label="W2V",
        zorder=3,
    )
    ax_metric.scatter(
        x_values,
        sbm_values,
        color=MODEL_COLORS["sbm"],
        marker=MODEL_MARKERS["sbm"],
        s=70,
        label="SBM",
        zorder=4,
    )
    ax_metric.scatter(
        x_values,
        random_values,
        color=MODEL_COLORS["random_k"],
        marker=MODEL_MARKERS["random_k"],
        s=70,
        label="Random",
        zorder=5,
    )

    for window in windows:
        x_value = x_map[window]

        if (
            window in top_w2v.index
            and pd.notna(top_w2v.loc[window].get("w2v+kmeans_self_metric"))
        ):
            y_value = float(top_w2v.loc[window, "w2v+kmeans_self_metric"])
            ax_metric.annotate(
                f"{y_value:.3f}",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(-8, -8),
                ha="right",
                va="top",
                fontsize=8,
                color=MODEL_COLORS["w2v+kmeans"],
            )

        if (
            window in top_sbm.index
            and pd.notna(top_sbm.loc[window].get("sbm_metric"))
        ):
            y_value = float(top_sbm.loc[window, "sbm_metric"])
            ax_metric.annotate(
                f"{y_value:.3f}",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(8, -5),
                ha="left",
                va="top",
                fontsize=8,
                color=MODEL_COLORS["sbm"],
            )

        if (
            window in top_random.index
            and pd.notna(top_random.loc[window].get("random_metric"))
        ):
            y_value = float(top_random.loc[window, "random_metric"])
            ax_metric.annotate(
                f"{y_value:.3f}",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(8, 8),
                ha="left",
                va="bottom",
                fontsize=8,
                color=MODEL_COLORS["random_k"],
            )

    limits = _metric_axis_limits([w2v_values, sbm_values, random_values])
    if limits is not None:
        ax_metric.set_ylim(*limits)

    ax_metric.set_ylabel(metric.upper())
    ax_metric.set_title(
        f"{metric.upper()} | Seed:{seed} | Samples:{number_of_samples} | "
        "W2V self-comparison vs best SBM and Random matches"
    )
    ax_metric.grid(True, axis="y", alpha=0.25)
    ax_metric.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_metric.spines["bottom"].set_visible(False)

    bottom_points = []

    sbm_df = top_sbm.reset_index().rename(columns={"index": "anchor_window"})
    sbm_df = sbm_df[sbm_df["sbm_window"].notna()].copy()
    for _, row in sbm_df.iterrows():
        anchor_window = str(row["anchor_window"])
        if anchor_window not in x_map:
            continue

        bottom_points.append(
            {
                "x": x_map[anchor_window] - 0.16,
                "window": str(row["sbm_window"]),
                "model": "sbm",
                "label": _window_token_label(
                    str(row["sbm_window"]),
                    is_context=is_context,
                ),
                "annotation_offset": (-8, 0),
                "ha": "right",
            }
        )

    random_df = top_random.reset_index().rename(columns={"index": "anchor_window"})
    random_df = random_df[random_df["random_window"].notna()].copy()
    for _, row in random_df.iterrows():
        anchor_window = str(row["anchor_window"])
        if anchor_window not in x_map:
            continue

        bottom_points.append(
            {
                "x": x_map[anchor_window] + 0.16,
                "window": str(row["random_window"]),
                "model": "random_k",
                "label": _window_token_label(
                    str(row["random_window"]),
                    is_context=is_context,
                ),
                "annotation_offset": (8, 0),
                "ha": "left",
            }
        )

    if bottom_points:
        unique_bottom_windows = sorted(
            {point["window"] for point in bottom_points},
            key=_window_sort_key,
        )
        y_map = {window: idx for idx, window in enumerate(unique_bottom_windows)}

        for model in ["sbm", "random_k"]:
            model_points = [point for point in bottom_points if point["model"] == model]
            if not model_points:
                continue

            ax_bottom.scatter(
                [point["x"] for point in model_points],
                [y_map[point["window"]] for point in model_points],
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                s=70,
                label=f"Best {_model_label(model)} {dimension_label}",
                zorder=5,
            )

            for point in model_points:
                ax_bottom.annotate(
                    point["label"],
                    (point["x"], y_map[point["window"]]),
                    textcoords="offset points",
                    xytext=point["annotation_offset"],
                    ha=point["ha"],
                    va="center",
                    fontsize=8,
                    color=MODEL_COLORS[model],
                )

        ax_bottom.set_yticks(range(len(unique_bottom_windows)))
        ax_bottom.set_yticklabels(
            [
                _window_token_label(window, is_context=is_context)
                for window in unique_bottom_windows
            ]
        )
        ax_bottom.set_ylim(len(unique_bottom_windows) - 0.5, -0.5)
    else:
        ax_bottom.text(
            0.5,
            0.5,
            "No SBM or Random cross-comparison rows found.",
            transform=ax_bottom.transAxes,
            ha="center",
            va="center",
        )

    ax_bottom.set_ylabel(f"Best matched {dimension_label}")
    ax_bottom.grid(True, axis="y", alpha=0.25)

    x_labels = [_window_token_label(window, is_context=is_context) for window in windows]
    ax_bottom.set_xticks(x_values)
    ax_bottom.set_xticklabels(x_labels)
    ax_bottom.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        pad=4,
    )
    ax_bottom.xaxis.set_label_position("top")
    ax_bottom.set_xlabel(f"W2V {dimension_label}", labelpad=8)
    ax_bottom.spines["top"].set_visible(True)
    ax_bottom.spines["top"].set_linewidth(1.2)

    handles_metric, labels_metric = ax_metric.get_legend_handles_labels()
    handles_bottom, labels_bottom = ax_bottom.get_legend_handles_labels()
    ax_metric.legend(
        handles_metric,
        labels_metric,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=3,
        frameon=False,
    )
    ax_bottom.legend(
        handles_bottom,
        labels_bottom,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )

    out_dir = output_dir / "w2vxsbmxrandom"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"dual_model_window_{metric.lower()}.png"

    fig.subplots_adjust(top=0.88, bottom=0.16, left=0.08, right=0.98, hspace=0.08)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved: {out_file}")
    return True


def plot_dual_model_window_metric(
    seed: int,
    number_of_samples: int,
    metric: str = "nmi",
):
    w2v_self_dir = _try_find_analysis_dir(
        DEFAULT_ANALYSES_ROOT,
        seed,
        number_of_samples,
        comparison_type="w2v_vs_w2v",
    )
    sbm_self_dir = _try_find_analysis_dir(
        DEFAULT_ANALYSES_ROOT,
        seed,
        number_of_samples,
        comparison_type="sbm_vs_sbm",
    )
    w2v_sbm_dir = _try_find_analysis_dir(
        DEFAULT_ANALYSES_ROOT,
        seed,
        number_of_samples,
        comparison_type="sbm_vs_w2v",
    )
    w2v_random_dir = _try_find_analysis_dir(
        DEFAULT_ANALYSES_ROOT,
        seed,
        number_of_samples,
        comparison_type="w2v_vs_random_k",
    )
    sbm_random_dir = _try_find_analysis_dir(
        DEFAULT_ANALYSES_ROOT,
        seed,
        number_of_samples,
        comparison_type="sbm_vs_random_k",
    )

    print(f"[INFO] Seed/samples matched: seed={seed}, samples={number_of_samples}")
    for label, path in [
        ("W2V self", w2v_self_dir),
        ("SBM self", sbm_self_dir),
        ("W2V × SBM", w2v_sbm_dir),
        ("W2V × Random", w2v_random_dir),
        ("SBM × Random", sbm_random_dir),
    ]:
        print(f"[INFO] Selected {label} analysis: {path.name if path else 'missing'}")

    is_context = any(
        _load_is_context(path)
        for path in [
            w2v_self_dir,
            sbm_self_dir,
            w2v_sbm_dir,
            w2v_random_dir,
            sbm_random_dir,
        ]
    )

    df_w2v_self = _summarize_self(
        _collect_self_rows(w2v_self_dir, metric, "w2v_vs_w2v", "w2v+kmeans"),
        "w2v+kmeans_self_metric",
    )
    df_sbm_self = _summarize_self(
        _collect_self_rows(sbm_self_dir, metric, "sbm_vs_sbm", "sbm"),
        "sbm_self_metric",
    )

    df_w2v_sbm = _summarize_best_cross(
        _collect_cross_rows(
            w2v_sbm_dir,
            metric,
            "sbm_vs_w2v",
            anchor_model="w2v+kmeans",
            candidate_model="sbm",
        ),
        candidate_window_column="sbm_window",
        metric_column="sbm_metric",
        is_context=is_context,
    )

    df_w2v_random = _summarize_best_cross(
        _collect_cross_rows(
            w2v_random_dir,
            metric,
            "w2v_vs_random_k",
            anchor_model="w2v+kmeans",
            candidate_model="random_k",
        ),
        candidate_window_column="random_window",
        metric_column="random_metric",
        is_context=is_context,
    )

    df_sbm_random = _summarize_best_cross(
        _collect_cross_rows(
            sbm_random_dir,
            metric,
            "sbm_vs_random_k",
            anchor_model="sbm",
            candidate_model="random_k",
        ),
        candidate_window_column="random_window",
        metric_column="random_metric",
        is_context=is_context,
    )

    w2v_sbm_output_root = w2v_sbm_dir or DEFAULT_ANALYSES_ROOT
    w2v_random_output_root = w2v_random_dir or DEFAULT_ANALYSES_ROOT
    sbm_random_output_root = sbm_random_dir or DEFAULT_ANALYSES_ROOT
    combined_output_root = w2v_sbm_dir or w2v_random_dir or DEFAULT_ANALYSES_ROOT

    made_any = False

    if not df_w2v_self.empty or not df_w2v_sbm.empty:
        made_any |= _plot_anchor_candidate(
            seed=seed,
            number_of_samples=number_of_samples,
            metric=metric,
            anchor_model="w2v+kmeans",
            candidate_model="sbm",
            df_anchor_self=df_w2v_self,
            df_candidate_best=df_w2v_sbm,
            candidate_window_column="sbm_window",
            candidate_metric_column="sbm_metric",
            output_dir=w2v_sbm_output_root,
            folder_name="w2vxsbm",
            is_context=is_context,
        )

    if not df_w2v_random.empty:
        made_any |= _plot_anchor_candidate(
            seed=seed,
            number_of_samples=number_of_samples,
            metric=metric,
            anchor_model="w2v+kmeans",
            candidate_model="random_k",
            df_anchor_self=df_w2v_self,
            df_candidate_best=df_w2v_random,
            candidate_window_column="random_window",
            candidate_metric_column="random_metric",
            output_dir=w2v_random_output_root,
            folder_name="w2vxrandom",
            is_context=is_context,
        )
    else:
        print("[INFO] Skipping w2vxrandom: missing W2V × Random data.")

    if not df_sbm_random.empty:
        made_any |= _plot_anchor_candidate(
            seed=seed,
            number_of_samples=number_of_samples,
            metric=metric,
            anchor_model="sbm",
            candidate_model="random_k",
            df_anchor_self=df_sbm_self,
            df_candidate_best=df_sbm_random,
            candidate_window_column="random_window",
            candidate_metric_column="random_metric",
            output_dir=sbm_random_output_root,
            folder_name="sbmxrandom",
            is_context=is_context,
        )
    else:
        print("[INFO] Skipping sbmxrandom: missing SBM × Random data.")

    if not df_w2v_sbm.empty and not df_w2v_random.empty:
        made_any |= _plot_w2v_sbm_random(
            seed=seed,
            number_of_samples=number_of_samples,
            metric=metric,
            df_w2v_self=df_w2v_self,
            df_sbm_best=df_w2v_sbm,
            df_random_best=df_w2v_random,
            output_dir=combined_output_root,
            is_context=is_context,
        )
    else:
        print("[INFO] Skipping w2vxsbmxrandom: missing W2V × SBM or W2V × Random data.")

    if not made_any:
        print(f"[WARN] No plot was generated for metric '{metric}'.")
        return False

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

    plot_dual_model_window_metric(args.seed, args.samples, args.metric)


if __name__ == "__main__":
    main()
