from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from graph_tool.all import (
    variation_information,
    partition_overlap,
    mutual_information,
    reduced_mutual_information,
)
from sklearn.metrics import adjusted_rand_score


def _compare_metrics(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    return {
        "vi": variation_information(labels_a, labels_b, norm=False),
        "nvi": variation_information(labels_a, labels_b, norm=True),
        "po": partition_overlap(labels_a, labels_b, norm=False),
        "npo": partition_overlap(labels_a, labels_b, norm=True),
        "mi": mutual_information(
            labels_a, labels_b, norm=False, adjusted=False
        ),
        "ami": mutual_information(
            labels_a, labels_b, norm=False, adjusted=True
        ),
        "nmi": mutual_information(
            labels_a, labels_b, norm=True, adjusted=False
        ),
        "anmi": mutual_information(
            labels_a, labels_b, norm=True, adjusted=True
        ),
        "ari": adjusted_rand_score(labels_a, labels_b),
        "rmi": reduced_mutual_information(labels_a, labels_b, norm=False),
        "nrmi": reduced_mutual_information(labels_a, labels_b, norm=True),
    }


def _window_sort_key(w):
    return float("inf") if w == "full" else int(w)


def plot_detailed_heatmap(df: pd.DataFrame, metric: str, out_path: Path):
    pivot = df.pivot(index="run_x", columns="run_y", values=metric)
    plt.figure(figsize=(10, 8))
    vmin, vmax = (
        (0, 1)
        if metric in {"nvi", "nmi", "anmi", "ami", "ari", "rmi", "nrmi", "npo"}
        else (0, None)
    )
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="viridis", vmin=vmin, vmax=vmax
    )
    plt.title(f"{metric.upper()} — {out_path.parent.name}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compare_runs_detailed(
    part_x, part_y, model_x, model_y, window_x, window_y, out_dir
):
    dfs_x, dfs_y = [], []
    for pf in part_x:
        try:
            run_idx = int(pf.stem.split("_run")[1].split("_")[0])
        except ValueError:
            continue
        df = pd.read_parquet(pf)
        df["run"] = run_idx
        dfs_x.append(df)

    for pf in part_y:
        try:
            run_idx = int(pf.stem.split("_run")[1].split("_")[0])
        except ValueError:
            continue
        df = pd.read_parquet(pf)
        df["run"] = run_idx
        dfs_y.append(df)

    if not dfs_x or not dfs_y:
        print("[WARN] Sem dados suficientes para comparar.")
        return

    data_x = pd.concat(dfs_x, ignore_index=True)
    data_y = pd.concat(dfs_y, ignore_index=True)

    data_x = data_x[
        (data_x["model"] == model_x)
        & (data_x["window"].astype(str) == str(window_x))
    ]
    data_y = data_y[
        (data_y["model"] == model_y)
        & (data_y["window"].astype(str) == str(window_y))
    ]

    runs_x = sorted(data_x["run"].unique())
    runs_y = sorted(data_y["run"].unique())

    if not runs_x or not runs_y:
        print("[WARN] Não há runs após o filtro.")
        return

    rows = []
    for rx in runs_x:
        for ry in runs_y:
            # evita auto-comparação exata
            if (
                model_x == model_y
                and str(window_x) == str(window_y)
                and rx == ry
            ):
                continue

            df_rx = data_x[data_x["run"] == rx]
            df_ry = data_y[data_y["run"] == ry]

            # >>> MÉTRICAS APENAS SOBRE TERMOS (tipo==1) E term NÃO-NULO
            df_rx_terms = df_rx[
                (df_rx["tipo"] == 1) & (df_rx["term"].notna())
            ].set_index("term")
            df_ry_terms = df_ry[
                (df_ry["tipo"] == 1) & (df_ry["term"].notna())
            ].set_index("term")

            common = df_rx_terms.index.intersection(df_ry_terms.index)
            if len(common) == 0:
                continue

            labels_x = df_rx_terms.loc[common]["label"].values
            labels_y = df_ry_terms.loc[common]["label"].values

            metrics = _compare_metrics(labels_x, labels_y)
            row = {"run_x": rx, "run_y": ry}
            row.update(metrics)
            rows.append(row)

    if not rows:
        print(
            f"[WARN] Sem comparações válidas entre {model_x}_J{window_x} e {model_y}_J{window_y}."
        )
        return

    df_result = pd.DataFrame(rows)
    df_result.to_parquet(out_dir / "metrics.parquet")

    for metric in rows[0].keys():
        if metric in {"run_x", "run_y"}:
            continue
        plot_detailed_heatmap(df_result, metric, out_dir / f"{metric}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, required=True)
    parser.add_argument("--seed", type=str, required=True)
    parser.add_argument(
        "--config",
        type=int,
        default=1,
        help="Número da CONFIG (ex: 1 para config_001)",
    )
    parser.add_argument(
        "--models", nargs="+", choices=["sbm", "w2v"], required=True
    )
    parser.add_argument("--window_x", type=str, required=True)
    parser.add_argument("--window_y", type=str, required=True)
    args = parser.parse_args()

    if len(args.models) == 1:
        model_x = model_y = args.models[0]
    elif len(args.models) == 2:
        model_x, model_y = args.models
    else:
        raise ValueError(
            "--models deve conter 1 ou 2 valores: ex: sbm ou sbm w2v"
        )

    # NOVA ESTRUTURA COM CONFIG
    base = (
        Path("../outputs/partitions")
        / args.samples
        / f"seed_{args.seed}"
        / f"config_{args.config:03d}"
    )
    part_x = sorted(
        (base / f"{model_x}_J{args.window_x}").glob("partitions_run*.parquet")
    )
    part_y = sorted(
        (base / f"{model_y}_J{args.window_y}").glob("partitions_run*.parquet")
    )

    out_dir = (
        base
        / "analysis_detailed"
        / f"{model_x}_J{args.window_x}_vs_{model_y}_J{args.window_y}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    compare_runs_detailed(
        part_x, part_y, model_x, model_y, args.window_x, args.window_y, out_dir
    )


if __name__ == "__main__":
    main()
