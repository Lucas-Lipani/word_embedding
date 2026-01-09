"""
Plota heatmaps de MÉDIAS de métricas agregadas para as 3 comparações:
- SBM vs W2V
- SBM vs SBM
- W2V vs W2V

Carrega dados PRÉ-CALCULADOS de conf/NNNN/analysis/{comparacao}/running_mean.parquet
NOVA ESTRUTURA: conf/NNNN/analysis/{model_x}_vs_{model_y}/running_mean.parquet
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _window_sort_key(w):
    """Ordena janelas de forma que 'full' fique no final."""
    return float("inf") if w == "full" else int(w)


def plot_average_heatmaps(config_dir: Path, model_x: str, model_y: str):
    """
    Plota heatmaps para MÉDIAS de {model_x} vs {model_y}.

    NOVA ESTRUTURA: conf/NNNN/analysis/{model_x}_vs_{model_y}/running_mean.parquet

    :param config_dir: Caminho da pasta conf/NNNN
    :param model_x: Modelo X (ex: "sbm", "w2v")
    :param model_y: Modelo Y (ex: "sbm", "w2v")
    """

    # Caminho do arquivo de métricas MÉDIAS
    metrics_file = (
        config_dir
        / "analysis"
        / f"{model_x}_vs_{model_y}"
        / "running_mean.parquet"
    )

    if not metrics_file.exists():
        print(f"[WARN] Arquivo não encontrado: {metrics_file}")
        return False

    try:
        df = pd.read_parquet(metrics_file)
    except Exception as e:
        print(f"[ERROR] Falha ao carregar parquet: {e}", file=sys.stderr)
        return False

    if df.empty:
        print(f"[WARN] DataFrame vazio: {metrics_file}", file=sys.stderr)
        return False

    print(f"\n[LOAD] {model_x.upper()} vs {model_y.upper()}: {df.shape}")

    # Identificar colunas de janela conforme a comparação
    if model_x == model_y:
        # Comparação homogênea: _row_window e _col_window
        row_key = f"{model_x}_row_window"
        col_key = f"{model_y}_col_window"
    else:
        # Comparação heterogênea: _window direto
        row_key = f"{model_x}_window"
        col_key = f"{model_y}_window"

    if row_key not in df.columns or col_key not in df.columns:
        print(
            f"[WARN] Colunas esperadas não encontradas para {model_x} vs {model_y}",
            file=sys.stderr,
        )
        print(f"[HINT] Encontrado: {df.columns.tolist()}", file=sys.stderr)
        return False

    # Derivar métricas disponíveis
    metric_cols = [c for c in df.columns if c not in {row_key, col_key}]

    if not metric_cols:
        print(f"[WARN] Nenhuma métrica encontrada", file=sys.stderr)
        return False

    # Diretório de saída
    out_dir = config_dir / "analysis" / f"{model_x}_vs_{model_y}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PLOT] Plotando {len(metric_cols)} métricas...")

    # Plotar cada métrica
    success_count = 0
    for metric in metric_cols:
        try:
            # Criar pivot table (MÉDIA JÁ CALCULADA)
            pivot = df.pivot(index=row_key, columns=col_key, values=metric)

            # Ordenar janelas
            rows_sorted = sorted(pivot.index.unique(), key=_window_sort_key)
            cols_sorted = sorted(pivot.columns.unique(), key=_window_sort_key)
            pivot = pivot.loc[rows_sorted, cols_sorted]

            # Definir escala apropriada
            if metric in {
                "nvi",
                "nmi",
                "anmi",
                "ami",
                "ari",
                "rmi",
                "nrmi",
                "npo",
            }:
                vmin, vmax = 0, 1
                cmap = "viridis"
            elif metric == "vi":
                vmin, vmax = None, None
                cmap = "RdYlGn_r"  # Vermelho (alto) para Verde (baixo)
            else:
                vmin, vmax = None, None
                cmap = "viridis"

            # Plotar heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={"label": metric.upper()},
                linewidths=0.5,
            )

            plt.title(
                f"{metric.upper()}: {model_x.upper()} vs {model_y.upper()} (MÉDIA)"
            )
            plt.xlabel(f"{model_y.upper()} Window")
            plt.ylabel(f"{model_x.upper()} Window")
            plt.tight_layout()

            # Salvar
            out_file = out_dir / f"heatmap_{metric}_avg.png"
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"✓ Heatmap salvo: {out_file}")
            success_count += 1

        except Exception as e:
            print(f"✗ Erro ao plotar {metric}: {e}", file=sys.stderr)
            plt.close()
            continue

    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Plota heatmaps de MÉDIAS para todas as comparações (NOVA ESTRUTURA)."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=int,
        required=True,
        help="Número da CONFIG (ex: 1 para conf/0001)",
    )
    args = parser.parse_args()

    # >>> NOVA ESTRUTURA: conf/NNNN
    base = Path("../outputs/conf")

    if not base.exists():
        print(
            f"[ERROR] Diretório base não encontrado: {base}", file=sys.stderr
        )
        print(
            f"[HINT] Execute primeiro: python3 -m word_embedding.window_experiments",
            file=sys.stderr,
        )
        sys.exit(1)

    config_dir = base / f"{args.config:04d}"

    if not config_dir.exists():
        print(f"[ERROR] Config não encontrada: {config_dir}", file=sys.stderr)
        available = sorted([d.name for d in base.glob("????")])
        if available:
            print(
                f"[HINT] Configs disponíveis: {', '.join(available)}",
                file=sys.stderr,
            )
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Plotando heatmaps de MÉDIA para todas as comparações")
    print(f"Config: {config_dir.name}")
    print(f"{'='*70}")

    # >>> As 3 comparações
    comparisons = [
        ("sbm", "w2v"),
        ("sbm", "sbm"),
        ("w2v", "w2v"),
    ]

    all_success = True
    for model_x, model_y in comparisons:
        print(f"\n### {model_x.upper()} vs {model_y.upper()}")
        success = plot_average_heatmaps(config_dir, model_x, model_y)
        if not success:
            all_success = False

    if all_success:
        print(f"\n✓ Todos os heatmaps de MÉDIA gerados com sucesso!")
        sys.exit(0)
    else:
        print(f"\n✗ Alguns heatmaps falharam", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
