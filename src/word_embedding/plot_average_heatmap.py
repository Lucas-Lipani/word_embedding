"""
Plota heatmaps de MÉDIAS de métricas agregadas.

Carrega dados PRÉ-CALCULADOS de analyses/NNNN/results.parquet
Arquivo contém comparações entre configurations com colunas de janelas.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc


def _window_sort_key(w):
    """Ordena janelas de forma que 'full' fique no final."""
    try:
        return (0, int(w))
    except (ValueError, TypeError):
        return (1, str(w))


def plot_average_heatmaps(analysis_dir: Path):
    """
    Plota heatmaps para todas as métricas do arquivo results.parquet.

    Estrutura: analyses/NNNN/results.parquet
    Colunas: config_x, config_y, run_x, run_y, window_x, window_y, [métricas...]

    :param analysis_dir: Caminho da pasta analyses/NNNN
    """

    # Caminho do arquivo de resultados
    metrics_file = analysis_dir / "results.parquet"

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

    print(f"\n[LOAD] Análise {analysis_dir.name}: {df.shape}")

    # Identificar colunas de janela e métricas
    row_key = "window_x"
    col_key = "window_y"

    if row_key not in df.columns or col_key not in df.columns:
        print(
            f"[WARN] Colunas esperadas não encontradas",
            file=sys.stderr,
        )
        print(f"[HINT] Encontrado: {df.columns.tolist()}", file=sys.stderr)
        return False

    # Derivar métricas disponíveis (excluir colunas de identificação)
    exclude_cols = {
        "config_x",
        "config_y",
        "run_x",
        "run_y",
        row_key,
        col_key,
        "model_x",
        "model_y",
    }
    metric_cols = [c for c in df.columns if c not in exclude_cols]

    if not metric_cols:
        print(f"[WARN] Nenhuma métrica encontrada", file=sys.stderr)
        return False

    # Diretório de saída
    out_dir = analysis_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PLOT] Plotando {len(metric_cols)} métricas...")

    # Plotar cada métrica
    success_count = 0
    for metric in metric_cols:
        try:
            # Criar pivot table (MÉDIA JÁ CALCULADA)
            pivot = df.pivot_table(
                index=row_key, columns=col_key, values=metric, aggfunc="mean"
            )

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
                cmap = getattr(cc.cm, "bgy")
            elif metric == "vi":
                vmin, vmax = None, None
                cmap = "RdYlGn_r"  # Vermelho (alto) para Verde (baixo)
            else:
                vmin, vmax = None, None
                cmap = "viridis"

            # Plotar heatmap
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={"label": metric.upper()},
                linewidths=0.5,
            )

            ax.invert_yaxis()

            plt.title(f"{metric.upper()}: Análise {analysis_dir.name} (MÉDIA)")
            plt.xlabel("Window Y")
            plt.ylabel("Window X")
            plt.tight_layout()

            # Salvar
            out_file = out_dir / f"heatmap_{metric}.png"
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"✓ Heatmap salvo: {out_file.name}")
            success_count += 1

        except Exception as e:
            print(f"✗ Erro ao plotar {metric}: {e}", file=sys.stderr)
            plt.close()
            continue

    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Plota heatmaps de MÉDIAS para análises."
    )
    parser.add_argument(
        "--analysis",
        "-a",
        type=int,
        help="Número da análise (ex: 1 para 0001). Se não fornecido, processa todas.",
    )
    args = parser.parse_args()

    # >>> Estrutura: analyses/NNNN
    base = Path("../outputs/analyses")

    if not base.exists():
        print(
            f"[ERROR] Diretório base não encontrado: {base}", file=sys.stderr
        )
        print(
            f"[HINT] Execute primeiro: python3 -m word_embedding.compute_analysis",
            file=sys.stderr,
        )
        sys.exit(1)

    # Selecionar análise(s)
    if args.analysis:
        analysis_dir = base / f"{args.analysis:04d}"
        if not analysis_dir.exists():
            print(
                f"[ERROR] Análise não encontrada: {analysis_dir}",
                file=sys.stderr,
            )
            available = sorted([d.name for d in base.glob("????")])
            if available:
                print(
                    f"[HINT] Análises disponíveis: {', '.join(available)}",
                    file=sys.stderr,
                )
            sys.exit(1)
        analysis_dirs = [analysis_dir]
    else:
        analysis_dirs = sorted([d for d in base.glob("????")])

    if not analysis_dirs:
        print(
            f"[ERROR] Nenhuma análise encontrada em: {base}", file=sys.stderr
        )
        sys.exit(1)

    all_success = True
    for analysis_dir in analysis_dirs:
        print(f"\n[ANÁLISE] {analysis_dir.name}")
        success = plot_average_heatmaps(analysis_dir)
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
