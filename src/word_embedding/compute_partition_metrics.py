from pathlib import Path
import pandas as pd
import argparse
import numpy as np
from graph_tool.all import (
    variation_information,
    partition_overlap,
    mutual_information,
    reduced_mutual_information,
)
from sklearn.metrics import adjusted_rand_score


def _window_sort_key(w):
    """
    Ordena janelas de forma que 'full' fique no final.
    """
    return float("inf") if w == "full" else int(w)


def _compare_metrics(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """
    Compara duas séries de rótulos de partição e 
    retorna um dicionário de métricas.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            "As duas séries de rótulos devem ter o mesmo comprimento."
        )
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


def compute_seed(seed_dir: Path, model_x: str, model_y: str):
    # Diretório de saída
    out_root = seed_dir / "analysis" / f"{model_x}_vs_{model_y}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Carrega todos os partitions_run*.parquet deste seed
    part_files = list(seed_dir.rglob("partitions_run*.parquet"))
    if not part_files:
        print(f"  [WARN] Nenhum partitions_run* encontrado em {seed_dir}")
        return

    dfs = []
    for pf in part_files:
        try:
            run_idx = int(pf.stem.split("_run")[1].split("_")[0])
        except ValueError:
            continue
        df = pd.read_parquet(pf)
        df["run"] = run_idx
        dfs.append(df)

    if not dfs:
        print(f"  [WARN] Partitions mal nomeados em {seed_dir}")
        return

    data = pd.concat(dfs, ignore_index=True)
    data["window"] = data["window"].astype(str)

    runs = sorted(data["run"].unique())
    windows = sorted(data["window"].unique(), key=_window_sort_key)

    # Nomes para linhas/colunas do heatmap de médias
    row_key = (
        f"{model_x}_row_window" if model_x == model_y else f"{model_x}_window"
    )
    col_key = (
        f"{model_y}_col_window" if model_x == model_y else f"{model_y}_window"
    )

    rows = []

    # Compara todas as janelas e runs
    for wx in windows:
        for wy in windows:
            for r1 in runs:
                df_x = data[
                    (data["model"] == model_x)
                    & (data["window"] == wx)
                    & (data["run"] == r1)
                ]
                if df_x.empty:
                    continue

                for r2 in runs:
                    if model_x == model_y and r1 == r2:
                        continue

                    df_y = data[
                        (data["model"] == model_y)
                        & (data["window"] == wy)
                        & (data["run"] == r2)
                    ]
                    if df_y.empty:
                        continue

                    # >>> MÉTRICAS APENAS SOBRE TERMOS (tipo==1) E NÃO-NULO
                    df_x_terms = df_x[
                        (df_x["tipo"] == 1) & (df_x["term"].notna())
                    ]
                    df_y_terms = df_y[
                        (df_y["tipo"] == 1) & (df_y["term"].notna())
                    ]
                    if df_x_terms.empty or df_y_terms.empty:
                        continue

                    common = sorted(
                        set(df_x_terms["term"]).intersection(
                            df_y_terms["term"]
                        ),
                        key=str,
                    )
                    if not common:
                        continue

                    labels_x = (
                        df_x_terms.set_index("term")
                        .loc[common]["label"]
                        .values
                    )
                    labels_y = (
                        df_y_terms.set_index("term")
                        .loc[common]["label"]
                        .values
                    )

                    metrics = _compare_metrics(labels_x, labels_y)
                    row_data = {row_key: wx, col_key: wy}
                    row_data.update(metrics)
                    rows.append(row_data)

    if not rows:
        print(
            f"  [WARN] Nenhum termo comum para {model_x} vs {model_y} em {seed_dir.name}"
        )
        return

    # Deriva as chaves de métricas a partir da primeira linha
    metric_keys = [k for k in rows[0].keys() if k not in {row_key, col_key}]

    mean_df = (
        pd.DataFrame(rows)
        .groupby([row_key, col_key])[metric_keys]
        .mean()
        .reset_index()
    )
    mean_df.to_parquet(out_root / "running_mean.parquet", engine="pyarrow")
    print(f"    running_mean salvo: {out_root / 'running_mean.parquet'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute mean comparison metrics for SBM and W2V partitions."
    )
    parser.add_argument(
        "--seed",
        "-s",
        help="Nome ou número da pasta seed a processar (ex: 42 ou seed_42). "
        "Se omitido, todas as seeds serão processadas.",
        default=None,
    )
    args = parser.parse_args()

    base = Path("../outputs/partitions")
    comparisons = [("sbm", "w2v"), ("sbm", "sbm"), ("w2v", "w2v")]

    for sample_dir in sorted(base.glob("*")):
        if not sample_dir.is_dir():
            continue
        print(f"Amostras: {sample_dir.name}")

        # Decide quais seeds percorrer
        if args.seed is None:
            seeds_to_check = sorted(sample_dir.glob("seed_*"))
        else:
            # aceita "42" ou "seed_42"
            seed_name = (
                args.seed
                if str(args.seed).startswith("seed_")
                else f"seed_{args.seed}"
            )
            seeds_to_check = [sample_dir / seed_name]

        for seed_dir in seeds_to_check:
            if not seed_dir.is_dir():
                print(
                    f"  [WARN] Seed {seed_dir.name} não encontrada para {sample_dir.name}"
                )
                continue
            print(f"  Seed: {seed_dir.name}")
            for model_x, model_y in comparisons:
                print(f"    {model_x.upper()} × {model_y.upper()}")
                compute_seed(seed_dir, model_x, model_y)
