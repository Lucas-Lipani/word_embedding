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
    """Compara duas séries de rótulos de partição."""
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


def plot_detailed_heatmap(df: pd.DataFrame, metric: str, out_path: Path):
    """Plota heatmap de uma métrica."""
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
    plt.title(f"{metric.upper()}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compare_runs_detailed(
    config_dir: Path, model_x: str, model_y: str, window_x: str, window_y: str
):
    """
    Compara runs de dois modelos/janelas diferentes dentro de UMA config.
    NOVA ESTRUTURA: conf/NNNN/run/RRRR/partition.parquet
    """
    out_dir = (
        config_dir
        / "analysis_detailed"
        / f"{model_x}_J{window_x}_vs_{model_y}_J{window_y}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Carregar partitions de TODOS os runs
    run_dirs = sorted(config_dir.glob("run/????"))

    if not run_dirs:
        print(f"    [WARN] Nenhum run encontrado em {config_dir.name}")
        return

    dfs = []
    for run_dir in run_dirs:
        partition_file = run_dir / "partition.parquet"
        if not partition_file.exists():
            continue

        try:
            run_idx = int(run_dir.name)
        except ValueError:
            continue

        df = pd.read_parquet(partition_file)
        df["run"] = run_idx
        dfs.append(df)

    if not dfs:
        print(f"    [WARN] Nenhum parquet encontrado em {config_dir.name}")
        return

    data = pd.concat(dfs, ignore_index=True)
    data["window"] = data["window"].astype(str)

    # Filtrar por modelo e janela
    data_x = data[
        (data["model"] == model_x) & (data["window"] == str(window_x))
    ]
    data_y = data[
        (data["model"] == model_y) & (data["window"] == str(window_y))
    ]

    runs_x = sorted(data_x["run"].unique())
    runs_y = sorted(data_y["run"].unique())

    if not runs_x or not runs_y:
        print(
            f"    [WARN] Sem dados para {model_x}_J{window_x} vs {model_y}_J{window_y}"
        )
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

            # APENAS termos (tipo==1) com term não-nulo
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

            try:
                metrics = _compare_metrics(labels_x, labels_y)
                row = {"run_x": rx, "run_y": ry}
                row.update(metrics)
                rows.append(row)
            except ValueError:
                continue

    if not rows:
        print(
            f"    [WARN] Sem comparações válidas para {model_x}_J{window_x} vs {model_y}_J{window_y}"
        )
        return

    df_result = pd.DataFrame(rows)
    df_result.to_parquet(out_dir / "metrics.parquet")
    print(f"      metrics salvo: {out_dir / 'metrics.parquet'}")

    # Plotar heatmaps de cada métrica
    for metric in df_result.columns:
        if metric not in {"run_x", "run_y"}:
            plot_detailed_heatmap(df_result, metric, out_dir / f"{metric}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comparações detalhadas entre runs (NOVA ESTRUTURA)."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=int,
        required=True,
        help="Número da CONFIG (ex: 1)",
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
        raise ValueError("--models deve ter 1 ou 2 valores")

    # NOVA ESTRUTURA: conf/NNNN
    base = Path("../outputs/conf")

    if not base.exists():
        print(f"[ERROR] Base não encontrada: {base}")
        exit(1)

    config_dir = base / f"{args.config:04d}"

    if not config_dir.exists():
        print(f"[ERROR] Config não encontrada: {config_dir}")
        exit(1)

    print(f"Config: {config_dir.name}")
    print(
        f"  {model_x.upper()} J{args.window_x} × {model_y.upper()} J{args.window_y}"
    )
    compare_runs_detailed(
        config_dir, model_x, model_y, args.window_x, args.window_y
    )
