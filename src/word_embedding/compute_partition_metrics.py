from pathlib import Path
import pandas as pd
import argparse
import sys
import numpy as np
from graph_tool.all import (
    variation_information,
    partition_overlap,
    mutual_information,
    reduced_mutual_information,
)
from sklearn.metrics import adjusted_rand_score


def _window_sort_key(w):
    """Ordena janelas de forma que 'full' fique no final."""
    return float("inf") if w == "full" else int(w)


def _compare_metrics(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """Compara duas séries de rótulos de partição e retorna um dicionário de métricas."""
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"As duas séries de rótulos devem ter o mesmo comprimento. "
            f"Recebido: {len(labels_a)} vs {len(labels_b)}"
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


def compute_config(config_dir: Path, model_x: str, model_y: str):
    """
    Processa UMA CONFIG específica, comparando modelo_x vs modelo_y.

    NOVA ESTRUTURA: conf/NNNN/run/RRRR/partition.parquet

    :param config_dir: Caminho da pasta conf/NNNN
    :param model_x: Modelo a comparar ("sbm" ou "w2v")
    :param model_y: Modelo a comparar ("sbm" ou "w2v")
    """
    # Diretório de saída dentro da CONFIG
    out_root = config_dir / "analysis" / f"{model_x}_vs_{model_y}"
    out_root.mkdir(parents=True, exist_ok=True)

    # >>> NOVO: carregar partitions de conf/NNNN/run/RRRR/partition.parquet
    run_dirs = sorted(config_dir.glob("run/????"))

    if not run_dirs:
        print(f"    [WARN] Sem runs encontradas em {config_dir.name}")
        return

    # Carrega todos os parquets
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

    runs = sorted(data["run"].unique())
    windows = sorted(data["window"].unique(), key=_window_sort_key)

    # Nomes para linhas/colunas do heatmap
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

                    try:
                        metrics = _compare_metrics(labels_x, labels_y)
                        row_data = {row_key: wx, col_key: wy}
                        row_data.update(metrics)
                        rows.append(row_data)
                    except ValueError as e:
                        print(f"      [SKIP] {e}")
                        continue

    if not rows:
        print(
            f"    [WARN] Nenhum termo comum para {model_x} vs {model_y} em {config_dir.name}"
        )
        return

    # Deriva as chaves de métricas
    metric_keys = [k for k in rows[0].keys() if k not in {row_key, col_key}]

    mean_df = (
        pd.DataFrame(rows)
        .groupby([row_key, col_key])[metric_keys]
        .mean()
        .reset_index()
    )
    mean_df.to_parquet(out_root / "running_mean.parquet", engine="pyarrow")
    print(f"      running_mean salvo: {out_root / 'running_mean.parquet'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute mean comparison metrics per CONFIG (NOVA ESTRUTURA)."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=int,
        help="Número da CONFIG a processar (ex: 1 para config_0001)",
        required=True,
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=str,
        default=None,
        help="Seed (ex: 42 ou seed_42). Opcional, apenas para referência.",
    )
    args = parser.parse_args()

    # >>> NOVA ESTRUTURA: conf/NNNN/run/RRRR
    base = Path("../outputs/conf")

    # >>> VALIDAÇÃO: verificar se base existe
    if not base.exists():
        print(
            f"[ERROR] Diretório base não encontrado: {base}", file=sys.stderr
        )
        print(
            f"[HINT] Execute primeiro: python3 -m word_embedding.window_experiments",
            file=sys.stderr,
        )
        sys.exit(1)

    # Encontrar a config certa
    config_dir = base / f"{args.config:04d}"

    # >>> VALIDAÇÃO: verificar se config existe
    if not config_dir.exists():
        print(f"[ERROR] Config não encontrada: {config_dir}", file=sys.stderr)
        available = sorted([d.name for d in base.glob("????")])
        if available:
            print(
                f"[HINT] Configs disponíveis: {', '.join(available)}",
                file=sys.stderr,
            )
        else:
            print(
                f"[HINT] Nenhuma config encontrada em {base}", file=sys.stderr
            )
        sys.exit(1)

    # Verificar se config tem runs
    run_dirs = list(config_dir.glob("run/????"))

    # >>> VALIDAÇÃO: verificar se há runs
    if not run_dirs:
        print(
            f"[ERROR] Nenhum run encontrado em {config_dir.name}",
            file=sys.stderr,
        )
        print(
            f"[HINT] Execute: python3 -m word_embedding.window_experiments",
            file=sys.stderr,
        )
        sys.exit(1)

    # >>> NOVO: Validar seed se fornecida
    if args.seed is not None:
        seed_name = (
            args.seed
            if str(args.seed).startswith("seed_")
            else f"seed_{args.seed}"
        )
        # Verificar se seed está contida em algum config.json
        found_seed = False
        try:
            import json

            config_file = config_dir / "config.json"
            if config_file.exists():
                with open(config_file, "r") as f:
                    cfg = json.load(f)
                    saved_seed = cfg.get("corpus", {}).get("seed")
                    if saved_seed is not None:
                        try:
                            if int(saved_seed) == int(args.seed):
                                found_seed = True
                        except (ValueError, TypeError):
                            pass

            if not found_seed:
                print(
                    f"[ERROR] Seed {args.seed} não encontrada em config {args.config:04d}",
                    file=sys.stderr,
                )
                if config_file.exists():
                    with open(config_file, "r") as f:
                        cfg = json.load(f)
                        saved_seed = cfg.get("corpus", {}).get("seed")
                        print(
                            f"[HINT] Config {args.config:04d} usa seed: {saved_seed}",
                            file=sys.stderr,
                        )
                sys.exit(1)
        except Exception as e:
            print(f"[WARN] Erro ao validar seed: {e}", file=sys.stderr)

    print(f"Config: {config_dir.name}")
    if args.seed:
        print(f"Seed: {args.seed}")

    comparisons = [("sbm", "w2v"), ("sbm", "sbm"), ("w2v", "w2v")]

    for model_x, model_y in comparisons:
        print(f"  {model_x.upper()} × {model_y.upper()}")
        compute_config(config_dir, model_x, model_y)
