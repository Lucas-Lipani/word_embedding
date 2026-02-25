from pathlib import Path
import time
import argparse
from collections import defaultdict, Counter
import sys

import pandas as pd
import spacy

from . import (
    graph_build,
    graph_sbm,
    results_io,
    w2vec_kmeans,
)

BASE_DIR = Path(__file__).resolve().parent


def _get_vertex_blocks_map(state, level: int = 0):
    """
    Return a vertex->block PropertyMap for BlockState/LayeredBlockState/NestedBlockState.
    For nested states, 'level' selects which hierarchy level to read (0 = base/original vertices).
    """
    if hasattr(state, "get_bs"):
        bs = state.get_bs()
        if not bs:
            raise ValueError("NestedBlockState sem níveis (get_bs() vazio).")
        return bs[level]

    if hasattr(state, "get_blocks"):
        try:
            return state.get_blocks()
        except Exception:
            pass
    if hasattr(state, "b"):
        return state.b

    raise TypeError(f"Tipo de state não suportado: {type(state)}")


def count_connected_term_blocks(state, g):
    """
    Retorna:
    - Número de blocos conectados que contêm TERMOS (tipo==1)
    - Número de blocos conectados que contêm JANELAS (tipo==5)
    - Dict com contagem de blocos por tipo
    """
    blocks_map = _get_vertex_blocks_map(state)

    connected = {
        int(blocks_map[v])
        for v in g.vertices()
        if (v.out_degree() + v.in_degree()) > 0
    }

    blocks_by_type = defaultdict(set)

    for v in g.vertices():
        b = int(blocks_map[v])
        t = int(g.vp["tipo"][v])
        if b in connected:
            blocks_by_type[t].add(b)

    print("\n[Depuração] Blocos conectados por tipo:")
    for t, bls in sorted(blocks_by_type.items()):
        nome = {
            0: "Documento",
            1: "Termo",
            3: "Janela",
            4: "Contexto",
            5: "JanelaSlide",
        }.get(t, f"Tipo {t}")
        print(f"  - {nome:<10}: {len(bls)} blocos")

    term_blocks = len(blocks_by_type.get(1, set()))
    window_blocks = len(blocks_by_type.get(5, set()))

    return term_blocks, window_blocks, dict(blocks_by_type)


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


def word_embedding(
    df_docs,
    nlp,
    window_list,
    graph_type="Document-SlideWindow-Term",
    n_blocks=None,
    fixed_seed=None,
    nested=False,
):
    """
    Retorna também as informações do grafo, blocos e W2V.
    """
    w2v_models = {
        w: w2vec_kmeans.get_or_train_w2v_model({}, w, df_docs, nlp)
        for w in window_list
    }

    for sbm_window in window_list:
        print(f"\n### SBM janela = {sbm_window}")

        # Constrói grafos de acordo com graph_type
        if graph_type == "Document-SlideWindow-Term":
            g_full, g_sbm_input = graph_build.build_window_graph_and_sliding(
                df_docs, nlp, sbm_window
            )
        elif graph_type == "Document-Term":
            # TODO: implementar build_doc_term_graph
            g_full = graph_build.extract_doc_term_graph(
                graph_build.build_window_graph_and_sliding(
                    df_docs, nlp, sbm_window
                )[0]
            )
            g_sbm_input = g_full
        elif graph_type == "Document-Context-Window-Term":
            g_full, _ = graph_build.build_window_graph_and_sliding(
                df_docs, nlp, sbm_window
            )
            g_win_term = graph_build.extract_window_term_graph(g_full)
            g_sbm_input = graph_build.extract_context_window_term_graph(
                g_win_term
            )
        elif graph_type == "Document-Window-Term":
            g_full = graph_build.build_window_graph(
                graph_build.initialize_graph(), df_docs, nlp, sbm_window
            )
            g_sbm_input = graph_build.extract_window_term_graph(g_full)
        else:
            raise ValueError(f"graph_type desconhecido: {graph_type}")

        # Aplicar SBM no grafo apropriado
        state = graph_sbm.sbm(g_sbm_input, n_blocks=None, nested=nested)

        # >>> EXTRAIR ENTROPY DO SBM
        try:
            sbm_entropy = state.entropy()
            print(f"[SBM] Entropy: {sbm_entropy:.4f}")
        except Exception as e:
            print(f"[WARN] Erro ao extrair entropy: {e}")
            sbm_entropy = None

        # >>> NOVO: Contar vértices PRÉ-SBM (estrutura do grafo)
        vertices_pre_sbm = defaultdict(int)
        for v in g_sbm_input.vertices():
            vertices_pre_sbm[int(g_sbm_input.vp["tipo"][v])] += 1

        print(f"\n[INFO] Estrutura do grafo PRÉ-SBM:")
        for tipo in sorted(vertices_pre_sbm.keys()):
            tipo_names = {0: "Doc", 1: "Termo", 3: "Janela", 5: "JanelaSlide"}
            print(
                f"  {tipo_names.get(tipo, f'Tipo {tipo}')}: {vertices_pre_sbm[tipo]}"
            )

        # >>> NOVO: Contar blocos PÓS-SBM
        term_blocks, window_blocks, blocks_by_type_set = (
            count_connected_term_blocks(state, g_sbm_input)
        )

        # Converter blocks_by_type_set para contagem (quantos blocos por tipo)
        blocks_post_sbm = {
            tipo: len(block_set)
            for tipo, block_set in blocks_by_type_set.items()
        }

        if n_blocks is not None:
            k_blocks = n_blocks
            print(f"Usando número fixo de blocos (W2V k) = {k_blocks}")
        else:
            k_blocks = term_blocks

        # >>> NOVO: Extrair informações do W2V
        w2v_model = w2v_models[sbm_window]
        w2v_sg = 1  # Skip-gram (conforme train_word2vec)
        w2v_window = 10000 if sbm_window == "full" else int(sbm_window)
        w2v_vector_size = 100  # conforme train_word2vec

        # ====== Extrai labels do SBM no grafo de entrada ======
        blocks_vec = _get_vertex_blocks_map(state).a

        sbm_rows = []
        term_to_block = {}

        # >>> DEBUG: contar vértices por tipo no grafo_sbm_input
        vertices_by_tipo = defaultdict(int)
        for v in g_sbm_input.vertices():
            vertices_by_tipo[int(g_sbm_input.vp["tipo"][v])] += 1

        print(f"\n[DEBUG] Vértices em g_sbm_input por tipo:")
        for tipo, count in sorted(vertices_by_tipo.items()):
            tipo_names = {0: "Doc", 1: "Termo", 5: "JanelaSlide"}
            print(f"  Tipo {tipo} ({tipo_names.get(tipo, '?')}): {count}")

        for v in g_sbm_input.vertices():
            t = int(g_sbm_input.vp["tipo"][v])
            name = g_sbm_input.vp["name"][v]
            sbm_lbl = int(blocks_vec[int(v)])

            row = {
                "window": sbm_window,
                "model": "sbm",
                "vertex": name,
                "tipo": t,
                "label": sbm_lbl,
                "doc_id": None,
                "term": None,
            }

            if t == 1:
                term_to_block[name] = sbm_lbl
                row["term"] = name

            sbm_rows.append(row)

        print(f"[DEBUG] sbm_rows gerados: {len(sbm_rows)}")

        # ====== Documentos (tipo 0) — label por maioria ======
        doc_rows = []
        for v_doc in g_full.vertices():
            if int(g_full.vp["tipo"][v_doc]) != 0:
                continue

            term_labels = []
            for e_doc in v_doc.all_edges():
                neighbor = (
                    e_doc.target()
                    if e_doc.source() == v_doc
                    else e_doc.source()
                )
                if int(g_full.vp["tipo"][neighbor]) == 1:
                    tname = g_full.vp["name"][neighbor]
                    lbl = term_to_block.get(tname)
                    if lbl is not None:
                        term_labels.append(lbl)
                else:
                    for e_inter in neighbor.all_edges():
                        v_term = (
                            e_inter.target()
                            if e_inter.source() == neighbor
                            else e_inter.source()
                        )
                        if int(g_full.vp["tipo"][v_term]) == 1:
                            tname = g_full.vp["name"][v_term]
                            lbl = term_to_block.get(tname)
                            if lbl is not None:
                                term_labels.append(lbl)

            if term_labels:
                counts = Counter(term_labels)
                maxc = max(counts.values())
                candidates = [l for l, c in counts.items() if c == maxc]
                doc_label = min(candidates)
            else:
                doc_label = pd.NA

            if (
                "doc_id" in g_full.vp
                and g_full.vp["doc_id"][v_doc] is not None
            ):
                doc_id_str = str(g_full.vp["doc_id"][v_doc])
            else:
                doc_id_str = str(g_full.vp["name"][v_doc])

            doc_rows.append(
                {
                    "window": sbm_window,
                    "model": "sbm",
                    "vertex": doc_id_str,
                    "tipo": 0,
                    "label": doc_label,
                    "doc_id": doc_id_str,
                    "term": None,
                }
            )

        print(f"[DEBUG] doc_rows gerados: {len(doc_rows)}")
        sbm_rows.extend(doc_rows)

        # ====== W2V clustering (somente termos) ======
        g_dt = graph_build.extract_doc_term_graph(g_full)
        _ = w2vec_kmeans.cluster_terms(
            g_dt, w2v_models[sbm_window], n_clusters=k_blocks
        )

        w2v_rows = []
        for v in g_dt.vertices():
            if int(g_dt.vp["tipo"][v]) != 1:
                continue
            term = g_dt.vp["name"][v]
            label = int(g_dt.vp["cluster"][v])
            w2v_rows.append(
                {
                    "window": sbm_window,
                    "model": "w2v",
                    "term": term,
                    "vertex": term,
                    "tipo": 1,
                    "doc_id": None,
                    "label": label,
                }
            )

        print(f"[DEBUG] w2v_rows gerados: {len(w2v_rows)}")

        yield sbm_rows, w2v_rows, sbm_entropy, k_blocks, vertices_pre_sbm, blocks_post_sbm, term_blocks, window_blocks, k_blocks, w2v_sg, w2v_window, w2v_vector_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", "--run", "--r", type=int, default=1, help="Nº de repetições"
    )
    parser.add_argument(
        "--samples", type=int, default=10, help="Nº de documentos amostrados"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed fixa (usada para todas as runs)",
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=None,
        help="Número fixo de blocos para o W2V.",
    )
    parser.add_argument(
        "--nested",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Usa SBM em modo nested.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        default=[5, 10, 20, 40, "full"],
        help="Lista de janelas (ex.: --windows 5 20 40 full)",
    )
    parser.add_argument(
        "--graph-type",
        choices=[
            "Document-Window-Term",
            "Document-SlideWindow-Term",
            "Document-Context-Window-Term",
            "Document-Term",
        ],
        default="Document-SlideWindow-Term",
        help="Tipo de grafo a construir",
    )

    args = parser.parse_args()

    n_runs = args.runs
    n_samples = args.samples
    fixed_seed = (
        args.seed if args.seed is not None else int(time.time()) % 2**32
    )

    # >>> VALIDAÇÃO: seed deve estar entre 0 e 2^32-1
    MAX_SEED = 2**32 - 1
    if fixed_seed < 0 or fixed_seed > MAX_SEED:
        print(
            f"[ERROR] Seed {fixed_seed} está fora do intervalo [0, {MAX_SEED}]",
            file=sys.stderr,
        )
        print(
            f"[HINT] Use um seed entre 0 e {MAX_SEED}",
            file=sys.stderr,
        )
        sys.exit(1)

    def _parse_win(x: str):
        x = str(x).strip().lower()
        return "full" if x == "full" else int(x)

    WINDOW_LIST = [_parse_win(w) for w in args.windows]

    OUT_CONF = Path("../outputs/conf")

    # Carrega spaCy e dados
    nlp = spacy.load("en_core_web_sm")
    df_full = pd.read_parquet("../data_lucas_argentina169.zstd")
    df_docs = df_full.sample(n=n_samples, random_state=fixed_seed)
    df_docs = tokenize_abstracts(df_docs, nlp)

    for r in range(n_runs):
        print(f"\n=== Execução {r+1}/{n_runs} ===")
        print(f"Seed usada (fixa): {fixed_seed}")
        print(f"Janelas: {WINDOW_LIST}")
        print(f"Tipo de Grafo: {args.graph_type}")

        # >>> MODIFICADO: processar cada janela SEPARADAMENTE
        for (
            sbm_rows,
            w2v_rows,
            sbm_entropy,
            k_blocks,
            vertices_pre_sbm,
            blocks_post_sbm,
            term_blocks,
            window_blocks,
            w2v_n_clusters,
            w2v_sg,
            w2v_window,
            w2v_vector_size,
        ) in word_embedding(
            df_docs,
            nlp,
            WINDOW_LIST,
            graph_type=args.graph_type,
            n_blocks=args.n_blocks,
            fixed_seed=fixed_seed,
            nested=args.nested,
        ):
            partitions_rows = []
            partitions_rows.extend(sbm_rows)
            partitions_rows.extend(w2v_rows)

            partitions_df = pd.DataFrame(partitions_rows)

            # Normalização de tipos para parquet
            partitions_df["window"] = partitions_df["window"].astype(str)
            partitions_df["label"] = pd.to_numeric(
                partitions_df["label"], errors="coerce"
            ).astype("Int64")
            if "label_members" in partitions_df.columns:
                partitions_df["label_members"] = partitions_df[
                    "label_members"
                ].astype("string")

            # Calcular número de blocos de termos
            n_term_blocks = term_blocks if term_blocks else 0

            # >>> EXTRAIR WINDOW_SIZE DO DATAFRAME
            window_size = (
                partitions_df["window"].iloc[0]
                if len(partitions_df) > 0
                else "5"
            )

            config_idx, run_idx, partition_file = (
                results_io.save_partitions_by_config(
                    base_conf_dir=OUT_CONF,
                    n_samples=n_samples,
                    seed=fixed_seed,
                    graph_type=args.graph_type,
                    nested=args.nested,
                    n_blocks=args.n_blocks,
                    run_idx=r + 1,
                    partitions_df=partitions_df,
                    window_size=window_size,
                    sbm_entropy=sbm_entropy,
                    vertices_pre_sbm=(
                        dict(vertices_pre_sbm) if vertices_pre_sbm else None
                    ),
                    blocks_post_sbm=(
                        dict(blocks_post_sbm) if blocks_post_sbm else None
                    ),
                    term_blocks_count=term_blocks,
                    window_blocks_count=window_blocks,
                    w2v_n_clusters=w2v_n_clusters,
                )
            )

            # Extrair janela do parquet para log
            window_val = (
                partitions_df["window"].iloc[0]
                if len(partitions_df) > 0
                else "?"
            )
            print(
                f"[SAVED] Config {config_idx:04d} | Run {run_idx:04d} | Window {window_size} | {partition_file}"
            )


if __name__ == "__main__":
    main()
