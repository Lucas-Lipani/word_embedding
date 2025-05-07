"""
window_experiments.py – Experimentos de janela de contexto
-----------------------------------------------------------
Varie (1) o parâmetro *window* do Word2Vec **e** (2) o tamanho da janela usada
para construir o grafo bipartido empregado pelo SBM.  Para o SBM, cada
sub-janela (pseudo-documento) vira um vértice do tipo DOCUMENTO; se a janela
for maior ou igual ao comprimento do texto, usa-se o documento integral –
o comportamento actual.

Resultado: um CSV em **outputs/window_comparison_results.csv** com VI, MI, NMI
e Pureza para cada configuração.
"""

import time
import argparse
from itertools import product
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec
from tqdm import tqdm

from main import (
    initialize_graph,
    build_bipartite_graph,
    min_sbm_wew,
    count_term_blocks,
    cluster_terms,
    compare_partitions_sbm_word2vec,
    compute_cluster_purity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segment_tokens(tokens, window_size: int, step: int):
    """Gera listas de *window_size* tokens com passo *step* (pode sobrepor)."""
    if window_size >= len(tokens):  # janela >= documento → apenas o doc inteiro
        yield tokens
        return

    for start in range(0, len(tokens) - window_size + 1, step):
        yield tokens[start : start + window_size]


def make_window_corpus(df: pd.DataFrame, nlp, window_size: int, step: int):
    """Cria um *DataFrame* onde cada linha é uma sub-janela (pseudo-documento)."""
    rows = []
    for doc_id, abstract in df["abstract"].items():
        doc = nlp(abstract)
        tokens = [t.text for t in doc if not t.is_punct and not t.is_space]
        for w_idx, window in enumerate(_segment_tokens(tokens, window_size, step)):
            rows.append(
                {
                    "orig_doc": doc_id,
                    "window_id": f"{doc_id}_{w_idx}",
                    "abstract": " ".join(window),
                }
            )
    return pd.DataFrame(rows)


def _sentences_from_df(df: pd.DataFrame, nlp):
    """Lista de sentenças tokenizadas (para o treino do Word2Vec)."""
    sents = []
    for abstract in df["abstract"]:
        doc = nlp(abstract)
        sents.append([t.text.lower() for t in doc if not t.is_punct and not t.is_stop])
    return sents


# ---------------------------------------------------------------------------
# Experimento principal
# ---------------------------------------------------------------------------

def experiment(
    df: pd.DataFrame,
    nlp,
    window_sizes: list[int],
    step_ratio: float = 1.0,
    vector_size: int = 100,
    min_count: int = 2,
    sg: int = 0,
):
    """
    Para cada *window_size*:
      1. Cria corpus de sub-janelas (se < inteiro do doc).
      2. Constrói grafo bipartido → aplica SBM.
      3. Treina Word2Vec com o mesmo *window*.
      4. Compara partições (VI/MI/NMI) e Pureza.
    Retorna *DataFrame* com as métricas.
    """
    results = []
    for w in window_sizes:
        step = max(1, int(w * step_ratio))  # passo (tokens) entre janelas
        print(f"\n  Tamanho da janela = {w} tokens  |  passo = {step}")

        df_win = make_window_corpus(df, nlp, w, step) if w < 10**6 else df.copy()
        print(f"   → {len(df_win)} pseudo-docs gerados")

        # —— Grafo + SBM ——
        g = initialize_graph()
        g = build_bipartite_graph(g, df_win, nlp)
        state = min_sbm_wew(g)
        n_term_blocks = count_term_blocks(g, state)

        # —— Word2Vec ——
        sentences = _sentences_from_df(df_win, nlp)
        w2v = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=w,
            min_count=min_count,
            sg=sg,
            workers=4,
        )

        clusters = cluster_terms(g, w2v, n_term_blocks)

        # —— Métricas ——
        vi, mi, po = compare_partitions_sbm_word2vec(g, state)
        nmi = po[2] if isinstance(po, tuple) else po

        purity_df = compute_cluster_purity(clusters, state, g)
        mean_purity = purity_df["Pureza"].mean() if not purity_df.empty else 0.0

        results.append(
            {
                "window": w,
                "num_pseudo_docs": len(df_win),
                "VI": vi,
                "MI": mi,
                "NMI": nmi,
                "mean_purity": mean_purity,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compara Word2Vec x SBM variando janela de contexto.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="wos_sts_journals.parquet",
        help="Arquivo Parquet com a coluna 'abstract'.",
    )
    parser.add_argument(
        "--windows",
        default="20,50,100,1000000",
        help="Lista de tamanhos de janela (tokens) separados por vírgula. Use um número bem grande para representar 'documento inteiro'.",
    )
    parser.add_argument("--vector_size", type=int, default=100)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--sg", type=int, choices=[0, 1], default=0, help="0=CBOW, 1=Skip-  gram")
    args = parser.parse_args()

    t0 = time.time()
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_parquet(args.input).sample(n=300, random_state=42)

    window_sizes = [int(x) for x in args.windows.split(",")]

    df_res = experiment(
        df,
        nlp,
        window_sizes,
        vector_size=args.vector_size,
        min_count=args.min_count,
        sg=args.sg,
    )

    out_path = "outputs/window_comparison_results.csv"
    df_res.to_csv(out_path, index=False)
    print("\nResultados salvos em", out_path)
    print(df_res)


    df = pd.read_csv("outputs/window_comparison_results.csv")

    max_w = df["window"].max()
    df["window_norm"] = df["window"] / max_w      # 0–1

    plt.figure(figsize=(8,6))
    plt.plot(df["window_norm"], df["NMI"],  marker='o', label='NMI')
    plt.plot(df["window_norm"], df["mean_purity"], marker='s', label='Mean purity')
    plt.plot([0,1],[0,1], '--', color='gray', label='linha ideal')

    plt.xlabel("Tamanho da janela (% do doc)")
    plt.ylabel("Similaridade")
    plt.title("SBM × Word2Vec vs tamanho da janela")
    plt.grid(True);  plt.legend();  plt.tight_layout()
    plt.savefig("outputs/window_comparison_plot.png", dpi=300)
    print("Gráfico salvo em outputs/window_comparison_plot.png")

    print(f"Tempo total: {time.time() - t0:.1f}s")
