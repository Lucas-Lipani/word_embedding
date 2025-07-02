from pathlib import Path
import time
import spacy
import pandas as pd
import argparse
from collections import defaultdict

from . import (
    graph_build,
    graph_sbm,
    graph_draw,
    results_io,
    w2vec_kmeans,
)

BASE_DIR = Path(__file__).resolve().parent


def count_connected_term_blocks(state, g):
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
        nome = {0: "Documento", 1: "Termo", 3: "Janela", 4: "Contexto"}.get(tipo, f"Tipo {tipo}")
        print(f"  - {nome:<10}: {len(blocos)} blocos")

    return len(term_blocks)


def word_embedding(df_docs, nlp, window_list):
    w2v_models = {
        w: w2vec_kmeans.get_or_train_w2v_model({}, w, df_docs, nlp)
        for w in window_list
    }

    sbm_term_labels = {}
    w2v_term_labels = {}

    for sbm_window in window_list:
        print(f"\n### SBM janela = {sbm_window}")
        g_full = graph_build.initialize_graph()
        g_full = graph_build.build_window_graph(g_full, df_docs, nlp, sbm_window)

        g_jan_term = graph_build.extract_window_term_graph(g_full)
        g_con_jan_term = graph_build.extract_context_window_term_graph(g_jan_term)
        doc_term = graph_build.extract_doc_term_graph(g_full)

        state = graph_sbm.sbm(g_con_jan_term)
        k_blocks = count_connected_term_blocks(state, g_con_jan_term)

        # SBM labels
        blocks_vec = state.get_blocks().a
        term_to_block = {
            g_jan_term.vp["name"][v]: int(blocks_vec[int(v)])
            for v in g_jan_term.vertices()
            if int(g_jan_term.vp["tipo"][v]) == 1
        }
        sbm_term_labels[sbm_window] = term_to_block.copy()

        # W2V clustering
        g_dt = doc_term.copy()
        _ = w2vec_kmeans.cluster_terms(g_dt, w2v_models[sbm_window], n_clusters=k_blocks)

        w2v_labels = {}
        for v in g_dt.vertices():
            if int(g_dt.vp["tipo"][v]) != 1:
                continue
            term = g_dt.vp["name"][v]
            label = int(g_dt.vp["cluster"][v])
            w2v_labels[term] = label
        w2v_term_labels[sbm_window] = list(w2v_labels.values())

    return sbm_term_labels, w2v_term_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10, help="Nº de repetições")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None, help="Seed fixa (usada para todas as runs)")
    args = parser.parse_args()

    n_runs = args.runs
    n_samples = args.samples
    fixed_seed = args.seed if args.seed is not None else int(time.time()) % 2**32

    WINDOW_LIST = [5, 10, 20, 40, "full"]
    OUT_PARTITIONS = BASE_DIR / "../../outputs/partitions"

    nlp = spacy.load("en_core_web_sm")
    df_full = pd.read_parquet("../wos_sts_journals.parquet")

    df_docs = df_full.sample(n=n_samples, random_state=fixed_seed)

    for r in range(n_runs):
        print(f"\n=== Execução {r+1}/{n_runs} ===")
        print(f"[ℹ] Seed usada (fixa): {fixed_seed}")

        sbm_term_labels, w2v_term_labels = word_embedding(df_docs, nlp, WINDOW_LIST)

        partitions_rows = []
        for w, term_map in sbm_term_labels.items():
            for term, label in term_map.items():
                partitions_rows.append({
                    "window": w, "model": "sbm", "term": term, "label": label
                })

        for w, labels in w2v_term_labels.items():
            terms = list(sbm_term_labels[w].keys())
            for term, label in zip(terms, labels):
                partitions_rows.append({
                    "window": w, "model": "w2v", "term": term, "label": label
                })

        partitions_df = pd.DataFrame(partitions_rows)
        partitions_df["window"] = partitions_df["window"].astype(str)

        # Salvar partições separadas por modelo e janela
        for model in ["sbm", "w2v"]:
            for window in WINDOW_LIST:
                df_model = partitions_df[
                    (partitions_df["model"] == model) &
                    (partitions_df["window"] == str(window))
                ]
                if not df_model.empty:
                    idx, file = results_io.save_partitions_only(
                        base_dir=OUT_PARTITIONS,
                        n_samples=n_samples,
                        seed=fixed_seed,
                        model_name=model,
                        window=window,
                        partitions_df=df_model
                    )
                    print(f"[✔] {model.upper()}_J{window} run {idx:03d} salvo em {file.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
