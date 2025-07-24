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
    Compara duas séries de rótulos de partição e retorna um dicionário de métricas.
    """
    # print(len(labels_a), len(labels_b))
    if len(labels_a) != len(labels_b):
        raise ValueError("As duas séries de rótulos devem ter o mesmo comprimento.")
    # else:
    #     # print("  [OK] Comprimentos iguais:", len(labels_a))
    #     continue
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


def compute_seed(seed_dir: Path, model_x: str, model_y: str):
    # Define o diretório onde o arquivo final "running_mean.parquet" será salvo
    out_root = seed_dir / "analysis" / f"{model_x}_vs_{model_y}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Coleta todos os arquivos de partições para as execuções (runs) deste seed
    part_files = list(seed_dir.rglob("partitions_run*.parquet"))
    if not part_files:
        print(f"  [WARN] Nenhum partitions_run* encontrado em {seed_dir}")
        return

    # Lê os dataframes de partição, adicionando o número da execução ("run") extraído do nome do arquivo
    dfs = []
    for pf in part_files:
        try:
            run_idx = int(pf.stem.split("_run")[1].split("_")[0])
        except ValueError:
            continue  # Ignora arquivos mal formatados
        df = pd.read_parquet(pf)
        df["run"] = run_idx
        dfs.append(df)

    if not dfs:
        print(f"  [WARN] Partitions mal nomeados em {seed_dir}")
        return

    # Junta todos os dataframes em um só
    data = pd.concat(dfs, ignore_index=True)
    data["window"] = data["window"].astype(
        str
    )  # garante que o tipo de janela seja string

    # Obtém a lista de execuções (runs) e janelas únicas, já ordenadas
    runs = sorted(data["run"].unique())
    windows = sorted(data["window"].unique(), key=_window_sort_key)

    # Define os nomes das colunas para linha/coluna do heatmap
    # Se os modelos forem iguais, cria "sbm_row_window" × "sbm_col_window"
    # Se forem diferentes, cria "sbm_window" × "w2v_window"
    row_key = f"{model_x}_row_window" if model_x == model_y else f"{model_x}_window"
    col_key = f"{model_y}_col_window" if model_x == model_y else f"{model_y}_window"

    rows = []  # aqui vamos armazenar os resultados de comparação

    # Loop sobre todas as janelas possíveis dos dois modelos
    for wx in windows:
        for wy in windows:
            # Loop sobre todas as execuções (runs) do modelo X
            for r1 in runs:
                # Filtra a partição de modelo X na janela wx e execução r1
                df_x = data[
                    (data["model"] == model_x)
                    & (data["window"] == wx)
                    & (data["run"] == r1)
                ]
                if df_x.empty:
                    continue

                # Loop sobre todas as execuções (runs) do modelo Y
                for r2 in runs:
                    # Se for o mesmo modelo e mesma execução, ignora (auto-comparação)
                    if model_x == model_y and r1 == r2:
                        continue

                    # Filtra a partição de modelo Y na janela wy e execução r2
                    df_y = data[
                        (data["model"] == model_y)
                        & (data["window"] == wy)
                        & (data["run"] == r2)
                    ]
                    if df_y.empty:
                        continue

                    # Verifica os termos em comum entre as duas partições
                    common = set(df_x["term"]).intersection(df_y["term"])

                    # Obtém os labels correspondentes aos termos em comum
                    labels_x = df_x.set_index("term").loc[list(common)]["label"].values
                    labels_y = df_y.set_index("term").loc[list(common)]["label"].values

                    metrics = _compare_metrics(labels_x, labels_y)
                    row_data = {row_key: wx, col_key: wy}
                    row_data.update(metrics)
                    rows.append(row_data)

    # Se nenhuma comparação foi válida (sem termos em comum suficientes), avisa e retorna
    if not rows:
        print(
            f"  [WARN] Nenhum termo comum para {model_x} vs {model_y} em {seed_dir.name}"
        )
        return

    # Converte os resultados em DataFrame e calcula a média por par de janelas
    metric_keys = list(metrics.keys())

    mean_df = (
        pd.DataFrame(rows).groupby([row_key, col_key])[metric_keys].mean().reset_index()
    )

    # Salva o arquivo de médias no formato Parquet
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

        # --- decide quais seeds percorrer ---
        if args.seed is None:
            seeds_to_check = sorted(sample_dir.glob("seed_*"))
        else:
            # aceita "42" ou "seed_42"
            seed_name = (
                args.seed if str(args.seed).startswith("seed_") else f"seed_{args.seed}"
            )
            seeds_to_check = [sample_dir / seed_name]

        # ------------------------------------
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
