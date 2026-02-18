"""
Plota heatmaps de MÉDIAS de métricas agregadas.

Carrega dados PRÉ-CALCULADOS de analyses/NNNN/results.parquet
Arquivo contém comparações entre configurations com colunas de janelas.
"""

import argparse
import json
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

    # Carregar config.json para obter metadados
    config_file = analysis_dir / "config.json"
    config_data = {}
    model_label = "Unknown"
    title_suffix = ""
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Extrair tipo de comparação (modelo)
            comparison_type = config_data.get("comparison_type", "unknown")
            model_label = comparison_type.replace("_", " ").title()
            
            # Extrair seed e número de samples para o título
            corpus = config_data.get("corpus", {})
            seed = corpus.get("seed", "unknown")
            num_samples = corpus.get("number_of_documents", "unknown")
            title_suffix = f"Seed: {seed} | Samples: {num_samples}"
        except Exception as e:
            print(f"[WARN] Erro ao carregar config.json: {e}", file=sys.stderr)

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

    # Extrair modelos dos eixos (se disponíveis)
    model_x_label = "Window X"
    model_y_label = "Window Y"
    if "model_x" in df.columns and "model_y" in df.columns:
        model_x = df["model_x"].iloc[0] if len(df) > 0 else "unknown"
        model_y = df["model_y"].iloc[0] if len(df) > 0 else "unknown"
        
        # Normalizar nomes de modelos
        def normalize_model_name(model):
            if model == "w2v+kmeans":
                return "W2V"
            else:
                return model.upper()
        
        model_x_norm = normalize_model_name(model_x)
        model_y_norm = normalize_model_name(model_y)
        model_x_label = f"Window {model_x_norm}"
        model_y_label = f"Window {model_y_norm}"

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

            plt.title(f"{metric.upper()}: {title_suffix}")
            plt.xlabel(model_y_label)
            plt.ylabel(model_x_label)
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
        nargs="*",
        help="Números das análises (ex: 1 3 5 para 0001, 0003, 0005). Se não fornecido, processa todas.",
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
        # Se foram passados números de análises
        analysis_dirs = []
        for analysis_num in args.analysis:
            analysis_dir = base / f"{analysis_num:04d}"
            if not analysis_dir.exists():
                print(
                    f"[ERROR] Análise não encontrada: {analysis_dir}",
                    file=sys.stderr,
                )
            else:
                analysis_dirs.append(analysis_dir)
        
        if not analysis_dirs:
            available = sorted([d.name for d in base.glob("????")])
            if available:
                print(
                    f"[HINT] Análises disponíveis: {', '.join(available)}",
                    file=sys.stderr,
                )
            sys.exit(1)
    else:
        # Se nenhum número foi passado, processa todas
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
