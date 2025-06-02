from pathlib import Path
import time
import spacy
import numpy as np
import pandas as pd
import matplotlib
from sklearn.metrics import adjusted_rand_score
import graph_build, graph_draw, w2vec_kmeans, graph_sbm, plots, compare_model
from collections import defaultdict


matplotlib.use("Agg")  # Usa backend para salvar arquivos, sem abrir janelas


def count_connected_term_blocks(state, g):
    """Retorna quantidade de blocos de termo (tipo 1) com vértices conectados.
    Também imprime, para depuração, a quantidade de blocos conectados por tipo.
    """
    blocks_vec = state.get_blocks().a

    # bloco é considerado ativo se tem vértices com arestas no grafo original
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

    # print de conferência
    print("\n[Depuração] Blocos conectados por tipo:")
    for tipo, blocos in sorted(blocks_by_type.items()):
        nome = {0: "Documento", 1: "Termo", 3: "Janela", 4: "Contexto"}.get(
            tipo, f"Tipo {tipo}"
        )
        print(f"  - {nome:<10}: {len(blocos)} blocos")

    return len(term_blocks)


def compare_partitions():
    pass


def compare_all_partitions(df, nlp, window_list): # AJUDA PRA NOMEAR ESSA FUNÇÃO PRINCIPAL
    results_vi = pd.DataFrame(index=window_list, columns=window_list)
    results_nmi = pd.DataFrame(index=window_list, columns=window_list)
    results_ari = pd.DataFrame(index=window_list, columns=window_list)

    w2v_models = {}
    sbm_term_labels = {}
    w2v_term_labels = {}

    for w_sbm in window_list:
        print(f"\n### SBM janela = {w_sbm}")

        # (a) construir grafo completo + SBM em JAN-TERM
        g_full = graph_build.initialize_graph()
        g_full = graph_build.build_window_graph(g_full, df, nlp, w_sbm)
        print("Grafo DOC-JAN-TERM:")
        print(g_full)

        g_jan_term = graph_build.extract_window_term_graph(g_full)
        print("Grafo JAN-TERM:")
        print(g_jan_term)

        g_con_jan_term = graph_build.extract_context_window_term_graph(g_jan_term)
        print("Grafo CONT-JAN-TERM")
        print(g_con_jan_term)

        doc_term = graph_build.extract_doc_term_graph(g_full)
        print("Grafo DOC-TERM:")
        print(doc_term)

        # # #  # Impressão dos 3 grafos bases do projeto
        # graph_draw.draw_base_graphs(g_full,g_jan_term,doc_term, g_con_jan_term, w_sbm)
        # exit()

        state = graph_sbm.sbm(g_con_jan_term)
        print("State do SBM:")
        print(state)

        # Definição da quantidade de clusters através do números de bocos de termos
        k_blocks = count_connected_term_blocks(state, g_con_jan_term)
        print(f"   \nblocos SBM (com termos e conexões) = {k_blocks}")

        blocks_vec = state.get_blocks().a
        term_to_block = {
            g_jan_term.vp["name"][v]: int(blocks_vec[int(v)])
            for v in g_jan_term.vertices()
            if int(g_jan_term.vp["tipo"][v]) == 1
        }

        sbm_term_labels[w_sbm] = list(term_to_block.values())

        for w_w2v in window_list:
            print(f"      → Word2Vec janela = {w_w2v}")

            if w_w2v not in w2v_models:
                w_int = 10000 if w_w2v == "full" else w_w2v
                w2v_models[w_w2v] = w2vec_kmeans.train_word2vec(df, nlp, w_int)
            w2v_model = w2v_models[w_w2v]

            g_dt = doc_term.copy()
            _ = w2vec_kmeans.cluster_terms(g_dt, w2v_model, n_clusters=k_blocks)

            sbm_labels = []
            w2v_labels = []
            for v in g_dt.vertices():
                if int(g_dt.vp["tipo"][v]) != 1:
                    continue
                term = g_dt.vp["name"][v]
                if term not in term_to_block:
                    continue
                sbm_labels.append(term_to_block[term])
                w2v_labels.append(int(g_dt.vp["cluster"][v]))

            if w_sbm == w_w2v:
                w2v_term_labels[w_w2v] = w2v_labels

            if len(set(w2v_labels)) > 1 and len(set(sbm_labels)) > 1:
                sbm_arr = np.array(sbm_labels)
                w2v_arr = np.array(w2v_labels)
                vi, mi, nmi = compare_model.compare_labels_multimetrics(sbm_arr, w2v_arr)
                ari = adjusted_rand_score(sbm_arr, w2v_arr)
            else:
                vi = nmi = ari = np.nan

            results_vi.loc[w_sbm, w_w2v] = vi
            results_nmi.loc[w_sbm, w_w2v] = nmi
            results_ari.loc[w_sbm, w_w2v] = ari

    compare_model.compare_same_model_partitions(sbm_term_labels, window_list, model_name="SBM")
    compare_model.compare_same_model_partitions(w2v_term_labels, window_list, model_name="Word2Vec")

    results_vi.to_csv("../../outputs/window/matriz_vi.csv")
    results_nmi.to_csv("../../outputs/window/matriz_nmi.csv")
    results_ari.to_csv("../../outputs/window/matriz_ari.csv")

    return results_vi, results_nmi, results_ari


def main():
    start = time.time()

    nlp = spacy.load("en_core_web_sm")
    df = pd.read_parquet("../../wos_sts_journals.parquet").sample(n=2, random_state=42)

    WINDOW_LIST = [5, "full"]

    vi_mat, nmi_mat, ari_mat = compare_all_partitions(df, nlp, WINDOW_LIST)

    # plot heatmaps
    plots.plot_clean_heatmap(
        nmi_mat, "NMI: SBM x Word2Vec", "../../outputs/window/cross_nmi.png", cmap="YlGnBu"
    )
    plots.plot_clean_heatmap(
        vi_mat,
        "VI: SBM x Word2Vec", 
        "../../outputs/window/cross_vi.png",
        cmap="YlOrBr",
        vmax=None,
    )
    plots.plot_clean_heatmap(
        ari_mat, "ARI: SBM x Word2Vec", "../../outputs/window/cross_ari.png", cmap="PuBuGn"
    )

    print(f"\nTempo total: {time.time() - start:.2f} s")


if __name__ == "__main__":
    main()
