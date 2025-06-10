from pathlib import Path
import time
import spacy
import pandas as pd
import os
from collections import defaultdict

from . import (
    graph_build,
    graph_draw,
    graph_sbm,
    plots,
    compare_model,
    w2vec_kmeans,
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def count_connected_term_blocks(state, g):
    """Retorna quantidade de blocos de termo (tipo 1) com vértices conectados.
    Também imprime, para depuração, a quantidade de blocos conectados por tipo.
    """
    blocks_vec = state.get_blocks().a

    connected_blocks = set()
    for v in g.vertices():
        if v.out_degree() + v.in_degree() > 0:
            bloco = int(blocks_vec[int(v)])
            connected_blocks.add(bloco)

    blocks_by_type = defaultdict(set)
    term_blocks = set()

    for v in g.vertices():
        tipo = int(g.vp["tipo"][v])
        bloco = int(blocks_vec[int(v)])
        if bloco in connected_blocks:
            blocks_by_type[tipo].add(bloco)
            if tipo == 1:
                term_blocks.add(bloco)

    print("\n[Depuração] Blocos conectados por tipo:")
    for tipo, blocos in sorted(blocks_by_type.items()):
        nome = {0: "Documento", 1: "Termo", 3: "Janela", 4: "Contexto"}.get(
            tipo, f"Tipo {tipo}"
        )
        print(f"  - {nome:<10}: {len(blocos)} blocos")

    return len(term_blocks)


def word_embedding(df_docs, nlp, window_list):

    w2v_models = {
        w: w2vec_kmeans.get_or_train_w2v_model({}, w, df_docs, nlp)
        for w in window_list
    }
    sbm_term_labels = {}
    sbm_term_labels_list = {}
    w2v_term_labels = {}

    for sbm_window in window_list:
        print(f"\n### SBM janela = {sbm_window}")

        g_full = graph_build.initialize_graph()
        g_full = graph_build.build_window_graph(
            g_full, df_docs, nlp, sbm_window
        )
        print("Grafo DOC-JAN-TERM:")
        print(g_full)

        g_jan_term = graph_build.extract_window_term_graph(g_full)
        print("Grafo JAN-TERM:")
        print(g_jan_term)

        g_con_jan_term = graph_build.extract_context_window_term_graph(
            g_jan_term
        )
        print("Grafo CONT-JAN-TERM")
        print(g_con_jan_term)

        doc_term = graph_build.extract_doc_term_graph(g_full)
        print("Grafo DOC-TERM:")
        print(doc_term)

        state = graph_sbm.sbm(g_con_jan_term)
        print(
            "State do SBM do grafo Termo como Contexto - Janela de Contexto - Termo:"
        )
        print(state)

        k_blocks = count_connected_term_blocks(state, g_con_jan_term)
        print(f"   \nblocos SBM (com termos e conexões) = {k_blocks}")

        results_vi, results_nmi, results_ari = (
            compare_model.compare_partitions(
                state,
                g_jan_term,
                sbm_term_labels,
                sbm_window,
                sbm_term_labels_list,
                w2v_models,
                window_list,
                doc_term,
                k_blocks,
                w2v_term_labels,
            )
        )

    compare_model.compare_same_model_partitions(
        sbm_term_labels_list, window_list, model_name="SBM"
    )
    compare_model.compare_same_model_partitions(
        w2v_term_labels, window_list, model_name="Word2Vec"
    )
    compare_model.compare_normal_sbm_partitions(
        doc_term, sbm_term_labels, window_list
    )

    output_dir = Path(__file__).resolve().parent / "../../outputs/window"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_vi.to_csv(output_dir / "matriz_vi.csv")
    results_nmi.to_csv(output_dir / "matriz_nmi.csv")
    results_ari.to_csv(output_dir / "matriz_ari.csv")

    return results_vi, results_nmi, results_ari


def main():
    start = time.time()

    nlp = spacy.load("en_core_web_sm")
    df_path = os.path.join(BASE_DIR, "../../wos_sts_journals.parquet")
    df_docs = pd.read_parquet(df_path).sample(n=3, random_state=42)

    WINDOW_LIST = [5, 10, 50, "full"]

    vi_mat, nmi_mat, ari_mat = word_embedding(df_docs, nlp, WINDOW_LIST)

    plots.plot_clean_heatmap(
        nmi_mat,
        "NMI: SBM x Word2Vec",
        os.path.join(BASE_DIR, "../../outputs/window/cross_nmi.png"),
        cmap="bgy",
    )
    plots.plot_clean_heatmap(
        vi_mat,
        "VI: SBM x Word2Vec",
        os.path.join(BASE_DIR, "../../outputs/window/cross_vi.png"),
        cmap="fire",
        vmax=None,
    )
    plots.plot_clean_heatmap(
        ari_mat,
        "ARI: SBM x Word2Vec",
        os.path.join(BASE_DIR, "../../outputs/window/cross_ari.png"),
        cmap="bgy",
    )

    print(f"\nTempo total: {time.time() - start:.2f} s")


if __name__ == "__main__":
    main()
