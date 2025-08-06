import matplotlib.pyplot as plt
import seaborn as sns
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
        "mi": mutual_information(labels_a, labels_b, norm=False, adjusted=False),
        "ami": mutual_information(labels_a, labels_b, norm=False, adjusted=True),
        "nmi": mutual_information(labels_a, labels_b, norm=True, adjusted=False),
        "anmi": mutual_information(labels_a, labels_b, norm=True, adjusted=True),
        "ari": adjusted_rand_score(labels_a, labels_b),
        "rmi": reduced_mutual_information(labels_a, labels_b, norm=False),
        "nrmi": reduced_mutual_information(labels_a, labels_b, norm=True),
    }


def _window_sort_key(w):
    return float("inf") if w == "full" else int(w)


def plot_detailed_heatmap(df: pd.DataFrame, metric: str, out_path: Path, model_x: str, model_y: str, window_x: str, window_y: str):

    label_x = f"{model_x}_J{window_x}"
    label_y = f"{model_y}_J{window_y}"

    df = df.copy()

    # Calcular número de partições por run
    # A função espera que 'data_x' e 'data_y' estejam disponíveis como argumentos,
    # mas como não estão, podemos incorporar isso no compare_runs_detailed e passar junto se necessário.
    # Aqui faremos a leitura dos arquivos novamente para fins de exemplo:
    def get_partition_counts(part_files):
        counts = {}
        for pf in part_files:
            try:
                run = int(pf.stem.split("_run")[1].split("_")[0])
                d = pd.read_parquet(pf)
                counts[run] = d["label"].nunique()
            except Exception:
                continue
        return counts

    # Reconstituir os caminhos dos arquivos para obter número de partições
    base = out_path.parent.parent.parent  # ../outputs/partitions/samples/seed_x
    part_x_dir = base / f"{model_x}_J{window_x}"
    part_y_dir = base / f"{model_y}_J{window_y}"
    part_x_files = list(part_x_dir.glob("partitions_run*.parquet"))
    part_y_files = list(part_y_dir.glob("partitions_run*.parquet"))

    partitions_x = get_partition_counts(part_x_files)
    partitions_y = get_partition_counts(part_y_files)

    # Montar rótulos do tipo "run - n_particoes"
    df["run_x_label"] = df["run_x"].apply(lambda x: f"{x} - {partitions_x.get(x, '?')}")
    df["run_y_label"] = df["run_y"].apply(lambda y: f"{y} - {partitions_y.get(y, '?')}")

    # Pivot e ordenação
    pivot = df.pivot(index="run_x_label", columns="run_y_label", values=metric)
    pivot = pivot.reindex(sorted(pivot.index, key=lambda x: int(x.split(" - ")[0])))  # Y crescente
    pivot = pivot.reindex(sorted(pivot.columns, key=lambda x: int(x.split(" - ")[0])), axis=1)  # X crescente

    # Plot
    plt.figure(figsize=(10, 8))
    vmin, vmax = (
        (0, 1) if metric in {"nvi", "nmi", "anmi", "ami", "ari", "rmi", "nrmi", "npo"} else (0, None)
    )
    ax = sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel(f"run {label_y}")
    ax.set_ylabel(f"run {label_x}")
    plt.title(f"{metric.upper()} — {label_x} vs {label_y}")
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

    data_x = pd.concat(dfs_x, ignore_index=True)
    data_y = pd.concat(dfs_y, ignore_index=True)

    data_x = data_x[
        (data_x["model"] == model_x) & (data_x["window"].astype(str) == str(window_x))
    ]
    data_y = data_y[
        (data_y["model"] == model_y) & (data_y["window"].astype(str) == str(window_y))
    ]

    runs_x = sorted(data_x["run"].unique())
    runs_y = sorted(data_y["run"].unique())

    rows = []
    for rx in runs_x:
        for ry in runs_y:
            df_rx = data_x[data_x["run"] == rx].set_index("term")
            df_ry = data_y[data_y["run"] == ry].set_index("term")

            if model_x == model_y and window_x == window_y and rx == ry:
                # Comparação da mesma run: inserir valores NaN
                metrics = {
                    "vi": np.nan, "nvi": np.nan, "po": np.nan, "npo": np.nan,
                    "mi": np.nan, "ami": np.nan, "nmi": np.nan, "anmi": np.nan,
                    "ari": np.nan, "rmi": np.nan, "nrmi": np.nan
                }

            else:
                common = df_rx.index.intersection(df_ry.index)
                if len(common) == 0:
                    continue

                labels_x = df_rx.loc[common]["label"].values
                labels_y = df_ry.loc[common]["label"].values
                metrics = _compare_metrics(labels_x, labels_y)

            row = {"run_x": rx, "run_y": ry}
            row.update(metrics)
            rows.append(row)


    if not rows:
        print(
            f"[WARN] Sem comparações válidas entre {model_x}_J{window_x} e {model_y}_J{window_y}"
        )
        return

    df_result = pd.DataFrame(rows)
    df_result.to_parquet(out_dir / "metrics.parquet")

    for metric in metrics.keys():
        plot_detailed_heatmap(df_result, metric, out_dir / f"{metric}.png", model_x, model_y, window_x, window_y)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, required=True)
    parser.add_argument("--seed", type=str, required=True)
    parser.add_argument("--models", nargs="+", choices=["sbm", "w2v"], required=True)
    parser.add_argument("--window_x", type=str, required=True)
    parser.add_argument("--window_y", type=str, required=True)
    args = parser.parse_args()

    if len(args.models) == 1:
        model_x = model_y = args.models[0]
    elif len(args.models) == 2:
        model_x, model_y = args.models
    else:
        raise ValueError("--models deve conter 1 ou 2 valores: ex: sbm ou sbm w2v")

    base = Path("../outputs/partitions") / args.samples / f"seed_{args.seed}"
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
