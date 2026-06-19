from pathlib import Path
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_ANALYSES_ROOT = Path("../outputs/analyses")


def _window_sort_key(value):
    text = str(value)
    if text.lower() == "full":
        return (1, text)
    try:
        return (0, int(text))
    except (ValueError, TypeError):
        return (0, text)


def _normalize_window_arg(value: str | None) -> str | None:
    """Normalize CLI window values such as '6t' to the stored value '6'."""
    if value is None:
        return None

    text = str(value).strip()
    if text.lower() == "full":
        return "full"

    if text.lower().endswith("t"):
        text = text[:-1]

    return text


def _window_label(value) -> str:
    text = str(value)
    if text.lower() == "full":
        return "Full"
    return f"{text}t"


def _load_comparison_type(analysis_dir: Path) -> str:
    config_file = analysis_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    comparison_type = config.get("comparison_type")
    if comparison_type is None:
        raise ValueError("Missing 'comparison_type' in config.json")

    return str(comparison_type)


def _load_results(analysis_dir: Path, metric: str) -> tuple[pd.DataFrame, str]:
    results_file = analysis_dir / "results.parquet"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    comparison_type = _load_comparison_type(analysis_dir)
    df = pd.read_parquet(results_file)

    required_columns = {"window_x", "window_y", metric}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in results.parquet: {missing_columns}")

    df = df.copy()
    df["window_x"] = df["window_x"].astype(str)
    df["window_y"] = df["window_y"].astype(str)
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])

    return df, comparison_type


def _filter_model_consistency(df: pd.DataFrame, comparison_type: str) -> pd.DataFrame:
    """Keep rows compatible with the selected comparison type without forcing the diagonal."""
    if {"model_x", "model_y"}.issubset(df.columns):
        if comparison_type in {"sbm_vs_sbm", "w2v_vs_w2v", "random_k_vs_random_k"}:
            df = df[df["model_x"] == df["model_y"]].copy()
        elif comparison_type in {"sbm_vs_w2v", "w2v_vs_random_k", "sbm_vs_random_k"}:
            df = df[df["model_x"] != df["model_y"]].copy()

    return df


def _filter_pair(df: pd.DataFrame, window_x: str, window_y: str, include_reverse: bool) -> pd.DataFrame:
    direct = (df["window_x"] == window_x) & (df["window_y"] == window_y)

    if include_reverse:
        reverse = (df["window_x"] == window_y) & (df["window_y"] == window_x)
        return df[direct | reverse].copy()

    return df[direct].copy()


def _filter_row(df: pd.DataFrame, row_window: str) -> pd.DataFrame:
    return df[df["window_x"] == row_window].copy()


def _filter_column(df: pd.DataFrame, column_window: str) -> pd.DataFrame:
    return df[df["window_y"] == column_window].copy()


def _add_jitter(n: int, scale: float = 0.035) -> pd.Series:
    if n <= 1:
        return pd.Series([0.0])
    return (pd.Series(range(n), dtype=float) - (n - 1) / 2) * scale


def _print_stats(df: pd.DataFrame, metric: str, group_cols: list[str]) -> None:
    if df.empty:
        return

    stats = (
        df.groupby(group_cols, dropna=False)[metric]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )

    print("\n[STATS]")
    print(stats.to_string(index=False))


def _plot_pair(
    df: pd.DataFrame,
    analysis_dir: Path,
    metric: str,
    comparison_type: str,
    window_x: str,
    window_y: str,
    include_reverse: bool,
) -> bool:
    if df.empty:
        print(f"[WARN] No rows found for pair window_x={window_x}, window_y={window_y}")
        return False

    fig, ax = plt.subplots(figsize=(7, 5))

    jitter = _add_jitter(len(df))
    ax.scatter(jitter, df[metric], alpha=0.75)

    mean_value = float(df[metric].mean())
    std_value = float(df[metric].std()) if len(df) > 1 else 0.0

    ax.errorbar(
        [0],
        [mean_value],
        yerr=[std_value],
        fmt="o",
        capsize=5,
    )
    ax.axhline(mean_value, linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xticks([0])
    ax.set_xticklabels([f"{_window_label(window_x)} × {_window_label(window_y)}"])
    ax.set_xlabel("Window pair")
    ax.set_ylabel(metric.upper())
    ax.set_title(
        f"{comparison_type} | {metric.upper()} distribution | "
        f"{_window_label(window_x)} × {_window_label(window_y)}"
    )
    ax.grid(True, axis="y", alpha=0.3)

    out_dir = analysis_dir / "scatter_by_window_pair"
    out_dir.mkdir(parents=True, exist_ok=True)

    reverse_suffix = "_with_reverse" if include_reverse else ""
    out_file = out_dir / f"scatter_{metric}_x{window_x}_y{window_y}{reverse_suffix}.png"

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close(fig)

    print(f"[INFO] Comparison type: {comparison_type}")
    print(f"[INFO] Number of points: {len(df)}")
    print(f"[INFO] Mean {metric}: {mean_value:.6f}")
    print(f"[INFO] Std  {metric}: {std_value:.6f}")
    print(f"[OK] Saved: {out_file}")
    _print_stats(df, metric, ["window_x", "window_y"])
    return True


def _plot_row_or_column(
    df: pd.DataFrame,
    analysis_dir: Path,
    metric: str,
    comparison_type: str,
    fixed_axis: str,
    fixed_window: str,
) -> bool:
    if df.empty:
        print(f"[WARN] No rows found for {fixed_axis}={fixed_window}")
        return False

    variable_col = "window_y" if fixed_axis == "window_x" else "window_x"
    variable_label = "Window Y" if variable_col == "window_y" else "Window X"

    windows = sorted(df[variable_col].astype(str).unique(), key=_window_sort_key)
    x_map = {window: index for index, window in enumerate(windows)}

    fig, ax = plt.subplots(figsize=(12, 6))

    for window in windows:
        sub = df[df[variable_col].astype(str) == window].copy()
        x0 = x_map[window]
        jitter = _add_jitter(len(sub))
        ax.scatter(x0 + jitter, sub[metric], alpha=0.7)

    stats = (
        df.groupby(variable_col, dropna=False)[metric]
        .agg(["mean", "std"])
        .reindex(windows)
    )
    stats["x"] = range(len(windows))

    ax.errorbar(
        stats["x"],
        stats["mean"],
        yerr=stats["std"].fillna(0),
        fmt="o",
        capsize=5,
    )

    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([_window_label(window) for window in windows])
    ax.set_xlabel(variable_label)
    ax.set_ylabel(metric.upper())

    fixed_label = _window_label(fixed_window)
    title_axis = "row window_x" if fixed_axis == "window_x" else "column window_y"
    ax.set_title(
        f"{comparison_type} | {metric.upper()} distribution by {variable_label} | "
        f"fixed {title_axis} = {fixed_label}"
    )
    ax.grid(True, axis="y", alpha=0.3)

    out_dir = analysis_dir / "scatter_by_window_line"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "row" if fixed_axis == "window_x" else "col"
    out_file = out_dir / f"scatter_{metric}_{suffix}_{fixed_window}.png"

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close(fig)

    print(f"[INFO] Comparison type: {comparison_type}")
    print(f"[INFO] Number of points: {len(df)}")
    print(f"[OK] Saved: {out_file}")
    _print_stats(df, metric, [fixed_axis, variable_col])
    return True


def plot_scatter(
    analysis_dir: Path,
    metric: str,
    window_x: str | None,
    window_y: str | None,
    row_window: str | None,
    column_window: str | None,
    include_reverse: bool,
) -> bool:
    df, comparison_type = _load_results(analysis_dir, metric)
    df = _filter_model_consistency(df, comparison_type)

    if df.empty:
        print(f"[WARN] No data found for comparison type: {comparison_type}")
        return False

    if window_x is not None or window_y is not None:
        if window_x is None or window_y is None:
            raise ValueError("Use --window-x and --window-y together.")
        selected = _filter_pair(df, window_x, window_y, include_reverse)
        return _plot_pair(
            selected,
            analysis_dir,
            metric,
            comparison_type,
            window_x,
            window_y,
            include_reverse,
        )

    if row_window is not None:
        selected = _filter_row(df, row_window)
        return _plot_row_or_column(
            selected,
            analysis_dir,
            metric,
            comparison_type,
            fixed_axis="window_x",
            fixed_window=row_window,
        )

    if column_window is not None:
        selected = _filter_column(df, column_window)
        return _plot_row_or_column(
            selected,
            analysis_dir,
            metric,
            comparison_type,
            fixed_axis="window_y",
            fixed_window=column_window,
        )

    raise ValueError(
        "Choose one mode: --window-x/--window-y for a cell, "
        "--row-window for a heatmap row, or --column-window for a heatmap column."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot run-level scatter for a specific window pair, row, or column."
    )
    parser.add_argument("--analysis", "-a", type=int, required=True)
    parser.add_argument("--metric", "-m", type=str, default="nmi")
    parser.add_argument("--window-x", type=str, default=None, help="Fixed window_x/cell row, e.g. 6 or 6t")
    parser.add_argument("--window-y", type=str, default=None, help="Fixed window_y/cell column, e.g. 7 or 7t")
    parser.add_argument("--row-window", type=str, default=None, help="Plot all window_y distributions for one window_x row")
    parser.add_argument("--column-window", type=str, default=None, help="Plot all window_x distributions for one window_y column")
    parser.add_argument(
        "--include-reverse",
        action="store_true",
        help="For pair mode, also include the symmetric cell window_y x window_x.",
    )
    parser.add_argument(
        "--analyses-root",
        type=Path,
        default=DEFAULT_ANALYSES_ROOT,
        help="Base analyses directory.",
    )

    args = parser.parse_args()

    analysis_dir = args.analyses_root / f"{args.analysis:04d}"

    plot_scatter(
        analysis_dir=analysis_dir,
        metric=args.metric,
        window_x=_normalize_window_arg(args.window_x),
        window_y=_normalize_window_arg(args.window_y),
        row_window=_normalize_window_arg(args.row_window),
        column_window=_normalize_window_arg(args.column_window),
        include_reverse=args.include_reverse,
    )


if __name__ == "__main__":
    main()
