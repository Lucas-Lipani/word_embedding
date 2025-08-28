from pathlib import Path
import argparse
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

    row_key = (
        f"{model_x}_row_window" if model_x == model_y else f"{model_x}_window"
    )
    col_key = (
        f"{model_y}_col_window" if model_x == model_y else f"{model_y}_window"
    )

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
        df[col_key],
        sorted(df[col_key].unique(), key=_window_sort_key),
        ordered=True,
    )
    pivot = df.pivot(index=row_key, columns=col_key, values=metric)

    plt.figure(figsize=(len(pivot.columns) + 1, len(pivot.index) + 1))
    normalized_metrics = {
        "nvi",
        "nmi",
        "anmi",
        "ami",
        "ari",
        "rmi",
        "nrmi",
        "npo",
    }
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

    ax = plt.gca()
    nrows, ncols = pivot.shape
    n = min(nrows, ncols)
    for i in range(n):
        # corrigindo pelo eixo y invertido
        ax.add_patch(
            plt.Rectangle(
                (i, nrows - 1 - i), 1, 1, fill=False, edgecolor="red", lw=2
            )
        )

    plt.title(f"Mean {metric.upper()} — {n_samples} samples")
    plt.ylabel(f"{model_x.upper()} window")
    plt.xlabel(f"{model_y.upper()} window")
    plt.tight_layout()

    out_png = parquet_file.parent / f"mean_{metric}.png"
    plt.savefig(out_png)
    plt.close()
    print(f"Heatmap salvo: {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Gera os heatmaps médios de métricas."
    )
    parser.add_argument(
        "--seed",
        type=str,  # ou int, se preferir
        help="processa apenas o diretório seed_X informado (ex.: 1754340049)",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("../outputs/partitions"),
        help="raiz das saídas de partição",
    )
    args = parser.parse_args()

    for sample_dir in sorted(args.base.glob("*")):
        if not sample_dir.is_dir():
            continue
        n_samples = sample_dir.name
        print(f"Amostras: {n_samples}")

        for seed_dir in sorted(sample_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue

            # --- filtro pelo argumento --seed -----------------
            if args.seed and seed_dir.name != f"seed_{args.seed}":
                continue
            # --------------------------------------------------

            print(f"  Seed: {seed_dir.name}")

            for pq in seed_dir.glob("analysis/*/running_mean.parquet"):
                df_prev = pd.read_parquet(pq, columns=None)
                metrics = set(df_prev.columns) - {
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
                for met in metrics:
                    plot_mean_heatmap(pq, met, n_samples)


if __name__ == "__main__":
    main()
