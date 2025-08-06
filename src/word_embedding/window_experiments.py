from pathlib import Path
import time
import spacy
import pandas as pd
import argparse
from collections import defaultdict
from graph_tool.all import graph_draw

from . import (
    graph_build,
    graph_sbm,
    # graph_draw,
    results_io,
    w2vec_kmeans,
)

BASE_DIR = Path(__file__).resolve().parent


def count_connected_term_blocks(state, g):
    blocks_map = state.get_blocks()
    connected = {
        int(blocks_map[v]) for v in g.vertices() if v.out_degree() + v.in_degree() > 0
    }

    blocks_by_type = defaultdict(set)
    term_blocks = set()

    for v in g.vertices():
        b = int(blocks_map[v])
        t = int(g.vp["tipo"][v])
        if b in connected:
            blocks_by_type[t].add(b)
            if t == 1:
                term_blocks.add(b)

    print("\n[Depuração] Blocos conectados por tipo:")
    for t, bls in sorted(blocks_by_type.items()):
        nome = {
            0: "Documento",
            1: "Termo",
            3: "Janela",
            4: "Contexto",
            5: "Jan-Slide",
        }.get(t, f"Tipo {t}")
        print(f"  - {nome:<10}: {len(bls)} blocos")

    return len(term_blocks)


def tokenize_abstracts(df, nlp):
    """
    Adiciona uma coluna 'tokens' ao DataFrame com a lista de tokens tratados.
    """
    print(f"Tokenizando {len(df)} abstracts com spaCy...")
    tokens_all = []
    for abstract in df["abstract"]:
        doc = nlp(abstract)
        tokens = [
            token.text.lower().strip()
            for token in doc
            if not token.is_stop and not token.is_punct
        ]
        tokens_all.append(tokens)

    df = df.copy()
    df["tokens"] = tokens_all
    return df


def word_embedding(df_docs, nlp, window_list, n_blocks=None):
    w2v_models = {
        w: w2vec_kmeans.get_or_train_w2v_model({}, w, df_docs, nlp) for w in window_list
    }

    sbm_term_labels = {}
    w2v_term_labels = {}

    for sbm_window in window_list:
        print(f"\n### SBM janela = {sbm_window}")
        g_full = graph_build.initialize_graph()
        g_new_jan_term = graph_build.initialize_graph()
        # g_full = graph_build.build_window_graph(g_full, df_docs, nlp, sbm_window)

        g_full, g_new_jan_term = graph_build.build_window_graph_and_sliding(
            df_docs, nlp, sbm_window
        )
        g_sbm_input = g_new_jan_term
        state = graph_sbm.sbm(g_sbm_input, n_blocks = n_blocks)

        doc_term = graph_build.extract_doc_term_graph(g_full)

        # g_jan_term = graph_build.extract_window_term_graph(g_full)
        # g_con_jan_term = graph_build.extract_context_window_term_graph(g_jan_term)
        # g_sbm_input = g_con_jan_term
        # state = graph_sbm.sbm(g_con_jan_term, n_blocks=n_blocks)

        # g_sbm_input = g_con_jan_term
        # state = graph_sbm.sbm_with_fixed_term_blocks(g_con_jan_term, n_blocks)


        k_blocks = count_connected_term_blocks(state, g_sbm_input)
        print("blocos de termos:", k_blocks)

        # if n_blocks is None:
        #     k_blocks = count_connected_term_blocks(state, g_con_jan_term)
        # else:
        #     k_blocks = n_blocks
        #     print(f"[Blocos fixos] Usando {n_blocks} blocos")

        # SBM labels
        blocks_vec = state.get_blocks().a
        term_to_block = {
            g_sbm_input.vp["name"][v]: int(blocks_vec[int(v)])
            for v in g_sbm_input.vertices()
            if int(g_sbm_input.vp["tipo"][v]) == 1
            and (v.in_degree() + v.out_degree() > 0)
        }

        sbm_term_labels[sbm_window] = term_to_block.copy()

        # W2V clustering
        g_dt = doc_term.copy()
        _ = w2vec_kmeans.cluster_terms(
            g_dt, w2v_models[sbm_window], n_clusters=k_blocks
        )

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
    parser.add_argument(
        "--samples", type=int, default=100, help="Nº de documentos amostrados"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed fixa (usada para todas as runs)"
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=None,
        help="Número fixo de blocos (SBM). Se omitido, o SBM decide automaticamente.",
    )
    args = parser.parse_args()

    n_runs = args.runs
    n_samples = args.samples
    fixed_seed = args.seed if args.seed is not None else int(time.time()) % 2**32

    WINDOW_LIST = [5, 10, 20, 40, 50, "full"]  # Janelas para SBM e W2V
    # WINDOW_LIST = ["full"]
    OUT_PARTITIONS = BASE_DIR / "../../outputs/partitions"

    nlp = spacy.load("en_core_web_sm")
    df_full = pd.read_parquet("../wos_sts_journals.parquet")
    df_docs = df_full.sample(n=n_samples, random_state=fixed_seed)
    df_docs = tokenize_abstracts(df_docs, nlp)

    for r in range(n_runs):
        print(f"\n=== Execução {r+1}/{n_runs} ===")
        print(f"Seed usada (fixa): {fixed_seed}")

        sbm_term_labels, w2v_term_labels = word_embedding(
            df_docs, nlp, WINDOW_LIST, n_blocks=args.n_blocks
        )

        partitions_rows = []
        for w, term_map in sbm_term_labels.items():
            for term, label in term_map.items():
                partitions_rows.append(
                    {"window": w, "model": "sbm", "term": term, "label": label}
                )

        for w, labels in w2v_term_labels.items():
            terms = list(sbm_term_labels[w].keys())
            for term, label in zip(terms, labels):
                partitions_rows.append(
                    {"window": w, "model": "w2v", "term": term, "label": label}
                )

        partitions_df = pd.DataFrame(partitions_rows)
        partitions_df["window"] = partitions_df["window"].astype(str)

        for model in ["sbm", "w2v"]:
            for window in WINDOW_LIST:
                df_model = partitions_df[
                    (partitions_df["model"] == model)
                    & (partitions_df["window"] == str(window))
                ]
                if not df_model.empty:
                    idx, file = results_io.save_partitions_only(
                        base_dir=OUT_PARTITIONS,
                        n_samples=n_samples,
                        seed=fixed_seed,
                        model_name=model,
                        window=window,
                        partitions_df=df_model,
                    )
                    print(
                        f"{model.upper()}_J{window} run {idx:03d} salvo em {file.relative_to(Path.cwd())}"
                    )


if __name__ == "__main__":
    main()
