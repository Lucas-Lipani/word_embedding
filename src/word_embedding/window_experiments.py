from pathlib import Path
import time
import argparse
from collections import defaultdict, Counter

import pandas as pd
import spacy
from graph_tool.all import graph_draw as gt_draw

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
    # Nested: pick the requested level and extract its blocks map
    if hasattr(state, "get_bs"):
        bs = state.get_bs()
        if not bs:
            raise ValueError("NestedBlockState sem níveis (get_bs() vazio).")
        # nível base particiona os vértices originais
        return bs[level]

    # Não-nested (BlockState/LayeredBlockState)
    if hasattr(state, "get_blocks"):
        try:
            return state.get_blocks()
        except Exception:
            pass
    if hasattr(state, "b"):
        return state.b

    raise TypeError(f"Tipo de state não suportado: {type(state)}")


def count_connected_term_blocks(state, g, seed=None):
    """
    Retorna a quantidade de blocos conectados que contêm TERMOS (tipo==1),
    apenas para log/diagnóstico. Aceita seed só para logging.
    """
    blocks_map = _get_vertex_blocks_map(state)

    # Conjunto de blocos que possuem ao menos 1 vértice com grau > 0
    connected = {
        int(blocks_map[v])
        for v in g.vertices()
        if (v.out_degree() + v.in_degree()) > 0
    }

    blocks_by_type = defaultdict(set)
    term_blocks = set()

    for v in g.vertices():
        b = int(blocks_map[v])
        t = int(g.vp["tipo"][v])
        if b in connected:
            blocks_by_type[t].add(b)
            if t == 1:  # Termo
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

    if seed is not None:
        print(f"[DEBUG] Seed passada para count_connected_term_blocks: {seed}")

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


def word_embedding(df_docs, nlp, window_list, n_blocks=None, fixed_seed=None, nested=False):
    """
    Para cada janela em window_list:
      - constrói g_full (Doc–Jan(3)–Term(1)) e g_slide (Jan-Slide(5)–Term(1));
      - roda SBM em g_slide;
      - salva linhas no parquet para Termos(1), Janelas(5) e Documentos(0).
    """
    # Treina (ou carrega) W2V por janela
    w2v_models = {
        w: w2vec_kmeans.get_or_train_w2v_model({}, w, df_docs, nlp)
        for w in window_list
    }

    for sbm_window in window_list:
        print(f"\n### SBM janela = {sbm_window}")
        # Constrói grafos
        g_full, g_slide = graph_build.build_window_graph_and_sliding(
            df_docs, nlp, sbm_window
        )

        # Entrada do SBM: grafo Jan-Slide(5)–Term(1)
        g_sbm_input = g_slide
        state = graph_sbm.sbm(g_sbm_input, n_blocks=None, nested=nested)

        # Info de conectividade p/ escolher k de W2V
        if n_blocks is not None:
            k_blocks = n_blocks
            print(f"Usando número fixo de blocos (W2V k) = {k_blocks}")
        else:
            k_blocks = count_connected_term_blocks(
                state, g_sbm_input, seed=fixed_seed
            )

        # ====== Extrai labels do SBM no grafo de entrada (Termos e Janelas) ======
        blocks_vec = _get_vertex_blocks_map(state).a


        sbm_rows = []
        term_to_block = {}

        for v in g_sbm_input.vertices():
            t = int(g_sbm_input.vp["tipo"][v])
            name = g_sbm_input.vp["name"][v]
            sbm_lbl = int(blocks_vec[int(v)])

            row = {
                "window": sbm_window,
                "model": "sbm",
                "vertex": name,
                "tipo": t,
                "label": sbm_lbl,  # manter numérico para todos
                "doc_id": None,
                "term": None,
            }

            if t == 1:
                # Termos: guardamos label numérico (usado nas métricas)
                term_to_block[name] = sbm_lbl
                row["term"] = name

            elif t == 5:
                # Janelas: adicionar membros (lista de termos) em 'label_members'
                if (
                    "win_terms" in g_sbm_input.vp
                    and g_sbm_input.vp["win_terms"][v]
                ):
                    members = list(g_sbm_input.vp["win_terms"][v])
                else:
                    members = [
                        g_sbm_input.vp["name"][u]
                        for u in v.out_neighbors()
                        if int(g_sbm_input.vp["tipo"][u]) == 1
                    ]
                row["label_members"] = ";".join(members)

                # doc_id, se disponível
                if "doc_id" in g_sbm_input.vp:
                    row["doc_id"] = g_sbm_input.vp["doc_id"][v]

            sbm_rows.append(row)

        # ====== Documentos (tipo 0) — label por maioria dos labels dos termos do doc ======
        doc_rows = []
        for v_doc in g_full.vertices():
            if int(g_full.vp["tipo"][v_doc]) != 0:
                continue

            term_labels = []
            # Doc(0) -> Jan(3) -> Term(1)
            for e_doc_win in v_doc.out_edges():
                v_win = e_doc_win.target()
                if int(g_full.vp["tipo"][v_win]) != 3:
                    continue

                for e_win_term in v_win.out_edges():
                    v_term = e_win_term.target()
                    if int(g_full.vp["tipo"][v_term]) != 1:
                        continue
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
                    "tipo": 0,  # DOCUMENTO
                    "label": doc_label,  # numérico (nullable)
                    "doc_id": doc_id_str,
                    "term": None,
                }
            )

        sbm_rows.extend(doc_rows)

        # ====== W2V clustering (somente termos) ======
        # Usa Doc–Term agregado (ou seu fluxo já existente)
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

        yield sbm_rows, w2v_rows  # devolve linhas para a etapa de salvamento


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", type=int, default=10, help="Nº de repetições"
    )
    parser.add_argument(
        "--samples", type=int, default=100, help="Nº de documentos amostrados"
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
        help="Número fixo de blocos para o W2V. Se omitido, o W2V decide baseado no SBM.",
    )
    parser.add_argument(
        "--nested",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Usa SBM em modo nested (layered). Use --no-nested para desativar."
    )
    parser.add_argument(
        "--window",
        "-w",
        type=str,
        default=None,
        help="Uma única janela (ex.: -w 10 ou -w full)",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        default=None,
        help="Lista de janelas (ex.: --windows 5 20 40 full)",
    )

    args = parser.parse_args()

    n_runs = args.runs
    n_samples = args.samples
    fixed_seed = (
        args.seed if args.seed is not None else int(time.time()) % 2**32
    )

    def _parse_win(x: str):
        x = str(x).strip().lower()
        return "full" if x == "full" else int(x)

    if args.windows is not None and len(args.windows) > 0:
        WINDOW_LIST = [_parse_win(w) for w in args.windows]
    elif args.window is not None:
        WINDOW_LIST = [_parse_win(args.window)]
    else:
        WINDOW_LIST = [5, 20, 40, "full"]

    OUT_PARTITIONS = Path("../outputs/partitions")

    # Carrega spaCy e dados
    nlp = spacy.load("en_core_web_sm")
    df_full = pd.read_parquet("../wos_sts_journals.parquet")
    df_docs = df_full.sample(n=n_samples, random_state=fixed_seed)
    df_docs = tokenize_abstracts(df_docs, nlp)

    for r in range(n_runs):
        print(f"\n=== Execução {r+1}/{n_runs} ===")
        print(f"Seed usada (fixa): {fixed_seed}")
        print(f"Janelas: {WINDOW_LIST}")

        partitions_rows = []
        for sbm_rows, w2v_rows in word_embedding(
            df_docs,
            nlp,
            WINDOW_LIST,
            n_blocks=args.n_blocks,
            fixed_seed=fixed_seed,
            nested=args.nested,
        ):
            partitions_rows.extend(sbm_rows)
            partitions_rows.extend(w2v_rows)

        partitions_df = pd.DataFrame(partitions_rows)

        # normalização de tipos para parquet
        partitions_df["window"] = partitions_df["window"].astype(str)
        partitions_df["label"] = pd.to_numeric(
            partitions_df["label"], errors="coerce"
        ).astype("Int64")
        if "label_members" in partitions_df.columns:
            partitions_df["label_members"] = partitions_df[
                "label_members"
            ].astype("string")

        # salva por modelo/janela
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
                        n_blocks=args.n_blocks,
                        nested=args.nested,
                        graph_type="Document-SlideWindow-Term",  # ← ALTERADO
                    )
                    print(
                        f"{model.upper()}_J{window} run {idx:03d} salvo em {file}"
                    )


if __name__ == "__main__":
    main()
