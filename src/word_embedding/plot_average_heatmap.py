from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc


def _window_sort_key(win):
    return float("inf") if win == "full" else int(win)


def plot_mean_heatmap(parquet_file: Path, metric: str, n_samples: str):
    if not parquet_file.exists():
        return

    df = pd.read_parquet(parquet_file)
    model_x, model_y = parquet_file.parent.name.split("_vs_")

    row_key = f"{model_x}_row_window" if model_x == model_y else f"{model_x}_window"
    col_key = f"{model_y}_col_window" if model_x == model_y else f"{model_y}_window"

    if {row_key, col_key, metric}.difference(df.columns):
        return

    df = df[[row_key, col_key, metric]].dropna()
    if df.empty:
        return

    df[row_key] = pd.Categorical(
        df[row_key],
        sorted(df[row_key].unique(), key=_window_sort_key)[::-1],
        ordered=True,
    )
    df[col_key] = pd.Categorical(
        df[col_key], sorted(df[col_key].unique(), key=_window_sort_key), ordered=True
    )
    pivot = df.pivot(index=row_key, columns=col_key, values=metric)

    plt.figure(figsize=(len(pivot.columns) + 1, len(pivot.index) + 1))
    normalized_metrics = {"nvi", "nmi", "anmi", "ami", "ari", "rmi", "nrmi", "npo"}
    vmin, vmax = (0, 1) if metric in normalized_metrics else (0, None)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=getattr(cc.cm, "bgy"),
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        cbar=True,
    )

    plt.title(f"Mean {metric.upper()} — {n_samples} samples")
    plt.ylabel(f"{model_x.upper()} window")
    plt.xlabel(f"{model_y.upper()} window")
    plt.tight_layout()

    out_png = parquet_file.parent / f"mean_{metric}.png"
    plt.savefig(out_png)
    plt.close()
    print(f"Heatmap salvo: {out_png}")


if __name__ == "__main__":
    base = Path("../outputs/partitions")

    for sample_dir in sorted(base.glob("*")):
        if not sample_dir.is_dir():
            continue
        n_samples = sample_dir.name
        print(f"Amostras: {n_samples}")

        for seed_dir in sorted(sample_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            print(f"  Seed: {seed_dir.name}")

            for parquet_file in seed_dir.glob("analysis/*/running_mean.parquet"):
                df_preview = pd.read_parquet(parquet_file, columns=None)
                metric_candidates = set(df_preview.columns) - {
                    "run",
                    "window",
                    "model",
                    "term",
                    "label",
                    "w2v_col_window",
                    "w2v_row_window",
                    "w2v_window",
                    "sbm_window",
                    "sbm_row_window",
                    "sbm_col_window",
                }
                # Remove também row_key e col_key se quiser
                for met in metric_candidates:
                    plot_mean_heatmap(parquet_file, met, n_samples)
