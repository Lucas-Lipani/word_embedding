from pathlib import Path
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def _window_sort_key(w):
    try:
        return (0, int(w))
    except (ValueError, TypeError):
        return (1, str(w))


def _load_comparison_type(analysis_dir: Path) -> str:
    config_file = analysis_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    comparison_type = config.get("comparison_type")
    if comparison_type is None:
        raise ValueError("Missing 'comparison_type' in config.json")

    return comparison_type


def _filter_dataframe(df: pd.DataFrame, comparison_type: str) -> pd.DataFrame:
    df = df[df["window_x"].astype(str) == df["window_y"].astype(str)].copy()

    if "config_x" in df.columns and "config_y" in df.columns:
        df = df[df["config_x"] == df["config_y"]].copy()

    if comparison_type in {"sbm_vs_sbm", "w2v_vs_w2v"}:
        if "model_x" in df.columns and "model_y" in df.columns:
            df = df[df["model_x"] == df["model_y"]].copy()

    elif comparison_type == "sbm_vs_w2v":
        if "model_x" in df.columns and "model_y" in df.columns:
            df = df[df["model_x"] != df["model_y"]].copy()

    else:
        raise ValueError(f"Unsupported comparison type: {comparison_type}")

    return df


def plot_scatter_by_window(analysis_dir: Path, metric: str = "nmi"):
    df = pd.read_parquet(analysis_dir / "results.parquet")
    comparison_type = _load_comparison_type(analysis_dir)

    df = _filter_dataframe(df, comparison_type)

    if df.empty:
        print(f"[WARN] No data found for comparison type: {comparison_type}")
        return

    df["window"] = df["window_x"].astype(str)

    windows = sorted(df["window"].unique(), key=_window_sort_key)
    x_map = {w: i for i, w in enumerate(windows)}

    fig, ax = plt.subplots(figsize=(10, 6))

    for w in windows:
        sub = df[df["window"] == w]
        x0 = x_map[w]

        if len(sub) > 1:
            jitter = (pd.Series(range(len(sub))) - (len(sub) - 1) / 2) * 0.05
        else:
            jitter = pd.Series([0.0])

        ax.scatter(
            x0 + jitter,
            sub[metric],
            alpha=0.7,
        )

    stats = df.groupby("window")[metric].agg(["mean", "std"])
    stats = stats.reindex(windows)
    stats["x"] = range(len(windows))

    ax.errorbar(
        stats["x"],
        stats["mean"],
        yerr=stats["std"].fillna(0),
        fmt="o",
        capsize=5,
    )

    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows)
    ax.set_xlabel("Window")
    ax.set_ylabel(metric.upper())

    filename = f"scatter_{metric}.png"
    ax.set_title(filename.replace(".png", "").capitalize())
    ax.grid(True, axis="y", alpha=0.3)

    out_dir = analysis_dir / "scatter_by_window"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / filename
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()

    print(f"[INFO] Comparison type: {comparison_type}")
    print(f"[INFO] Number of points: {len(df)}")
    print(f"[OK] Saved: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", "-a", type=int, required=True)
    parser.add_argument("--metric", "-m", type=str, default="nmi")
    args = parser.parse_args()

    analysis_dir = Path("../outputs/analyses") / f"{args.analysis:04d}"
    plot_scatter_by_window(analysis_dir, args.metric)


if __name__ == "__main__":
    main()