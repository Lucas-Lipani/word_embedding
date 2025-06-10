import pandas as pd
import numpy as np
import os
from pathlib import Path
from graph_tool.all import (
    variation_information,
    mutual_information,
    partition_overlap,
)
from sklearn.metrics import adjusted_rand_score

from . import plots, graph_sbm, window_experiments, w2vec_kmeans

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../outputs/window"))
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def compare_partitions(
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
):

    results_vi = pd.DataFrame(index=window_list, columns=window_list)
    results_nmi = pd.DataFrame(index=window_list, columns=window_list)
    results_ari = pd.DataFrame(index=window_list, columns=window_list)

    # 1. Obtém mapeamento termo → bloco do SBM
    blocks_vec = state.get_blocks().a
    term_to_block = {
        g_jan_term.vp["name"][v]: int(blocks_vec[int(v)])
        for v in g_jan_term.vertices()
        if int(g_jan_term.vp["tipo"][v]) == 1
    }

    # Salva para comparações posteriores
    sbm_term_labels[sbm_window] = term_to_block.copy()
    sbm_term_labels_list[sbm_window] = list(term_to_block.values())

    for w_w2v in window_list:
        print(f"      → Word2Vec janela = {w_w2v}")

        w2v_model = w2v_models[w_w2v]

        # Cria cópia do grafo doc-term e clusteriza
        g_dt = doc_term.copy()
        _ = w2vec_kmeans.cluster_terms(g_dt, w2v_model, n_clusters=k_blocks)

        # Monta vetores de rótulos
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

        # Salva rótulos para comparações entre janelas iguais
        if sbm_window == w_w2v:
            w2v_term_labels[w_w2v] = w2v_labels

        # Aplica métricas se houver mais de um cluster
        if len(set(w2v_labels)) > 1 and len(set(sbm_labels)) > 1:
            sbm_arr = np.array(sbm_labels)
            w2v_arr = np.array(w2v_labels)
            vi, mi, nmi = compare_labels_multimetrics(sbm_arr, w2v_arr)
            ari = adjusted_rand_score(sbm_arr, w2v_arr)
        else:
            vi = nmi = ari = np.nan

        results_vi.loc[sbm_window, w_w2v] = vi
        results_nmi.loc[sbm_window, w_w2v] = nmi
        results_ari.loc[sbm_window, w_w2v] = ari

        return results_vi, results_nmi, results_ari


def compare_same_model_partitions(model_outputs, window_list, model_name):
    results_vi = pd.DataFrame(index=window_list, columns=window_list)
    results_nmi = pd.DataFrame(index=window_list, columns=window_list)
    results_ari = pd.DataFrame(index=window_list, columns=window_list)

    for i in window_list:
        for j in window_list:
            if i not in model_outputs or j not in model_outputs:
                results_vi.loc[i, j] = results_nmi.loc[i, j] = results_ari.loc[
                    i, j
                ] = np.nan
                continue

            labels_i = model_outputs[i]
            labels_j = model_outputs[j]

            if len(labels_i) != len(labels_j):
                results_vi.loc[i, j] = results_nmi.loc[i, j] = results_ari.loc[
                    i, j
                ] = np.nan
                continue
            vi, mi, nmi = compare_labels_multimetrics(
                np.array(labels_i), np.array(labels_j)
            )
            ari = adjusted_rand_score(labels_i, labels_j)
            results_vi.loc[i, j] = vi
            results_nmi.loc[i, j] = nmi
            results_ari.loc[i, j] = ari

    window_dir = Path(__file__).resolve().parent / "../../outputs/window"
    window_dir.mkdir(parents=True, exist_ok=True)

    results_vi.to_csv(
        window_dir / f"{model_name.lower()}_vs_{model_name.lower()}_vi.csv"
    )
    results_nmi.to_csv(
        window_dir / f"{model_name.lower()}_vs_{model_name.lower()}_nmi.csv"
    )
    results_ari.to_csv(
        window_dir / f"{model_name.lower()}_vs_{model_name.lower()}_ari.csv"
    )

    plots.plot_clean_heatmap(
        results_nmi,
        f"NMI: {model_name} × {model_name}",
        window_dir / f"{model_name.lower()}_vs_{model_name.lower()}_nmi.png",
        cmap="bgy",
    )
    plots.plot_clean_heatmap(
        results_vi,
        f"VI: {model_name} × {model_name}",
        window_dir / f"{model_name.lower()}_vs_{model_name.lower()}_vi.png",
        cmap="fire",
        vmax=None,
    )
    plots.plot_clean_heatmap(
        results_ari,
        f"ARI: {model_name} × {model_name}",
        window_dir / f"{model_name.lower()}_vs_{model_name.lower()}_ari.png",
        cmap="bgy",
    )

    return results_vi, results_nmi, results_ari


def compare_labels_multimetrics(labels_left, labels_right):
    """Calcula VI, MI, NMI para dois vetores de rótulos numpy."""
    vi = variation_information(labels_left, labels_right)
    mi = mutual_information(labels_left, labels_right)
    po = partition_overlap(labels_left, labels_right)
    nmi = po[2] if isinstance(po, (tuple, list, np.ndarray)) else po

    return vi, mi, nmi


def compare_normal_sbm_partitions(doc_term_graph, sbm_term_dicts, window_list):
    """
    Compara o SBM fixo do grafo DOC-TERM com cada SBM obtido
    para as diferentes janelas (CONT-JAN-TERM).

    Retorna e salva matrizes (1 × |windows|) de VI, NMI e ARI.
    """

    # ── 1. SBM fixo (DOC-TERM) ────────────────────────────────────
    state_fixed = graph_sbm.sbm(doc_term_graph)
    print("State do SBM do grafo Documento - Termo:")
    print(state_fixed)
    blocks_fixed = state_fixed.get_blocks().a
    term_to_block_fixed = {
        doc_term_graph.vp["name"][v]: int(blocks_fixed[int(v)])
        for v in doc_term_graph.vertices()
        if int(doc_term_graph.vp["tipo"][v]) == 1
    }

    _ = window_experiments.count_connected_term_blocks(
        state_fixed, doc_term_graph
    )

    # ── 2. Matrizes de resultado: linha única “SBM-DOC-TERM” ─────
    idx = ["SBM-DOC-TERM"]
    vi_df = pd.DataFrame(index=idx, columns=window_list)
    nmi_df = pd.DataFrame(index=idx, columns=window_list)
    ari_df = pd.DataFrame(index=idx, columns=window_list)

    # ── 3. Loop sobre cada janela ─────────────────────────────────
    for w in window_list:
        # dicionário {termo: bloco_sbm_janela}
        term_to_block_win = sbm_term_dicts.get(w, {})

        common_terms = set(term_to_block_fixed) & set(term_to_block_win)
        if len(common_terms) < 2:
            vi = nmi = ari = np.nan
        else:
            fixed_labels = np.array(
                [term_to_block_fixed[t] for t in common_terms]
            )
            win_labels = np.array([term_to_block_win[t] for t in common_terms])

            vi, _, nmi = compare_labels_multimetrics(fixed_labels, win_labels)
            ari = adjusted_rand_score(fixed_labels, win_labels)

        vi_df.loc[idx[0], w] = vi
        nmi_df.loc[idx[0], w] = nmi
        ari_df.loc[idx[0], w] = ari

    # ── 4. Salvar CSVs + heat-maps ────────────────────────────────
    out_dir = Path("../outputs/window")
    out_dir.mkdir(parents=True, exist_ok=True)

    vi_df.to_csv(out_dir / "sbm_docterm_vs_windows_vi.csv")
    nmi_df.to_csv(out_dir / "sbm_docterm_vs_windows_nmi.csv")
    ari_df.to_csv(out_dir / "sbm_docterm_vs_windows_ari.csv")

    plots.plot_clean_heatmap(
        vi_df,
        "VI: SBM DOC-TERM × SBM-Janela",
        out_dir / "sbm_docterm_vs_windows_vi.png",
        cmap="fire",
        vmax=None,
    )
    plots.plot_clean_heatmap(
        nmi_df,
        "NMI: SBM DOC-TERM × SBM-Janela",
        out_dir / "sbm_docterm_vs_windows_nmi.png",
        cmap="bgy",
    )
    plots.plot_clean_heatmap(
        ari_df,
        "ARI: SBM DOC-TERM × SBM-Janela",
        out_dir / "sbm_docterm_vs_windows_ari.png",
        cmap="bgy",
    )

    return vi_df, nmi_df, ari_df
