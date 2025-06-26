from pathlib import Path
import time
import spacy
import pandas as pd
import argparse
import os
from collections import defaultdict

from . import (
    graph_build,
    graph_draw,
    graph_sbm,
    results_io,
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

    # Inicializa matrizes de comparação global
    all_vi = pd.DataFrame(index=window_list, columns=window_list)
    all_nmi = pd.DataFrame(index=window_list, columns=window_list)
    all_ari = pd.DataFrame(index=window_list, columns=window_list)

    for sbm_window in window_list:
        print(f"\n### SBM janela = {sbm_window}")

        g_full = graph_build.initialize_graph()
        g_full = graph_build.build_window_graph(g_full, df_docs, nlp, sbm_window)

        g_jan_term = graph_build.extract_window_term_graph(g_full)
        g_con_jan_term = graph_build.extract_context_window_term_graph(g_jan_term)
        doc_term = graph_build.extract_doc_term_graph(g_full)

        state = graph_sbm.sbm(g_con_jan_term)
        k_blocks = count_connected_term_blocks(state, g_con_jan_term)

        results_vi, results_nmi, results_ari = compare_model.compare_partitions(
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

        # Copia os resultados da linha atual para a matriz total
        all_vi.loc[sbm_window] = results_vi.loc[sbm_window]
        all_nmi.loc[sbm_window] = results_nmi.loc[sbm_window]
        all_ari.loc[sbm_window] = results_ari.loc[sbm_window]

    # Comparações intra-modelo e SBM-DOC-TERM
    compare_model.compare_same_model_partitions(sbm_term_labels_list, window_list, model_name="SBM")
    compare_model.compare_same_model_partitions(w2v_term_labels, window_list, model_name="Word2Vec")
    compare_model.compare_normal_sbm_partitions(doc_term, sbm_term_labels, window_list)

    # Salvar e plotar resultados finais
    output_dir = Path(__file__).resolve().parent / "../../outputs/window"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_vi.to_csv(output_dir / "matriz_vi.csv")
    all_nmi.to_csv(output_dir / "matriz_nmi.csv")
    all_ari.to_csv(output_dir / "matriz_ari.csv")

    plots.plot_clean_heatmap(
        all_nmi,
        "NMI: SBM x Word2Vec",
        output_dir / "cross_nmi.png",
        cmap="bgy",
    )
    plots.plot_clean_heatmap(
        all_vi,
        "VI: SBM x Word2Vec",
        output_dir / "cross_vi.png",
        cmap="fire",
        vmax=None,
    )
    plots.plot_clean_heatmap(
        all_ari,
        "ARI: SBM x Word2Vec",
        output_dir / "cross_ari.png",
        cmap="bgy",
    )

    return (all_vi, all_nmi, all_ari,
    sbm_term_labels, # {window: {term: block}}
    w2v_term_labels) # {window: list(labels)}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10, help="Nº de repetições")
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()

    n_runs = args.runs
    n_samples = args.samples

    WINDOW_LIST = [5, 7, 10, 15, 20, 30, 40, 50, "full"]
    OUT_BASE   = Path(__file__).resolve().parent / "../../outputs/window"

    nlp = spacy.load("en_core_web_sm")
    df_full = pd.read_parquet("../wos_sts_journals.parquet")

    for r in range(n_runs):
        print(f"\n=== Execução {r+1}/{n_runs} ===")
        df_docs = df_full.sample(n=n_samples, random_state=int(time.time()) % 2**32)

        # ----------------- seu pipeline normal -----------------
        vi_mat, nmi_mat, ari_mat, sbm_term_labels, w2v_term_labels = word_embedding(df_docs, nlp, WINDOW_LIST)

        # --------------- empilha tudo em long format -----------
        metrics_long = (
            vi_mat.stack()
                .rename("vi")
                .to_frame()
                .join(nmi_mat.stack().rename("nmi"))
                .join(ari_mat.stack().rename("ari"))
                .reset_index(names=["sbm_window", "w2v_window"])
        )


        # partições: seu código já contém  sbm_term_labels_list  e  w2v_term_labels
        # faça algo no estilo:
        partitions_rows = []

        # --- SBM labels ---
        for w, term_map in sbm_term_labels.items():
            for term, label in term_map.items():
                partitions_rows.append({
                    "window": w, "model": "sbm", "term": term, "label": label
                })

        # --- Word2Vec labels ---
        for w, labels in w2v_term_labels.items():
            terms = list(sbm_term_labels[w].keys())  # mesma ordem dos rótulos w2v
            for term, label in zip(terms, labels):
                partitions_rows.append({
                    "window": w, "model": "w2v", "term": term, "label": label
                })

        partitions_df = pd.DataFrame(partitions_rows)

        # -------------------------------------------------------
        partitions_df["window"] = partitions_df["window"].astype(str)

        metrics_long["sbm_window"] = metrics_long["sbm_window"].astype(str)
        metrics_long["w2v_window"] = metrics_long["w2v_window"].astype(str)

        run_idx, m_file, p_file = results_io.save_run(OUT_BASE, n_samples,
                                           WINDOW_LIST, metrics_long,
                                           partitions_df)

        print(f">>> Resultados salvos (execução #{run_idx:03d})\n"
              f"    métricas   → {m_file.relative_to(Path.cwd())}\n"
              f"    partições  → {p_file.relative_to(Path.cwd())}")

if __name__ == "__main__":
    main()