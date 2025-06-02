import pandas as pd
from pathlib import Path
import numpy as np
import plots, graph_sbm, window_experiments
from graph_tool.all import variation_information, mutual_information, partition_overlap
from sklearn.metrics import adjusted_rand_score


def compare_same_model_partitions(model_outputs, window_list, model_name):
    results_vi = pd.DataFrame(index=window_list, columns=window_list)
    results_nmi = pd.DataFrame(index=window_list, columns=window_list)
    results_ari = pd.DataFrame(index=window_list, columns=window_list)

    for i in window_list:
        for j in window_list:
            labels_i = model_outputs[i]
            labels_j = model_outputs[j]
            if len(labels_i) != len(labels_j):
                results_vi.loc[i, j] = results_nmi.loc[i, j] = results_ari.loc[i, j] = (
                    np.nan
                )
                continue
            vi, mi, nmi = compare_labels_multimetrics(
                np.array(labels_i), np.array(labels_j)
            )
            ari = adjusted_rand_score(labels_i, labels_j)
            results_vi.loc[i, j] = vi
            results_nmi.loc[i, j] = nmi
            results_ari.loc[i, j] = ari

    out_dir = Path("../../outputs")
    window_dir = out_dir / "window"

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
        cmap="YlGnBu",
    )
    plots.plot_clean_heatmap(
        results_vi,
        f"VI: {model_name} × {model_name}",
        window_dir / f"{model_name.lower()}_vs_{model_name.lower()}_vi.png",
        cmap="YlOrBr",
        vmax=None,
    )
    plots.plot_clean_heatmap(
        results_ari,
        f"ARI: {model_name} × {model_name}",
        window_dir / f"{model_name.lower()}_vs_{model_name.lower()}_ari.png",
        cmap="PuBuGn",
    )

    return results_vi, results_nmi, results_ari


def compare_labels_multimetrics(labels_left, labels_right):
    """Calcula VI, MI, NMI para dois vetores de rótulos numpy."""
    vi = variation_information(labels_left, labels_right)
    mi = mutual_information(labels_left, labels_right)
    po = partition_overlap(labels_left, labels_right)
    nmi = po[2] if isinstance(po, (tuple, list, np.ndarray)) else po

    return vi, mi, nmi

    # print("State do SBM do grafo Documento - Termo:")
    # print(state_sbm_fixed)


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

    _ = window_experiments.count_connected_term_blocks(state_fixed, doc_term_graph)

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
            fixed_labels = np.array([term_to_block_fixed[t] for t in common_terms])
            win_labels = np.array([term_to_block_win[t] for t in common_terms])

            vi, _, nmi = compare_labels_multimetrics(fixed_labels, win_labels)
            ari = adjusted_rand_score(fixed_labels, win_labels)

        vi_df.loc[idx[0], w] = vi
        nmi_df.loc[idx[0], w] = nmi
        ari_df.loc[idx[0], w] = ari

    # ── 4. Salvar CSVs + heat-maps ────────────────────────────────
    out_dir = Path("../../outputs/window")
    out_dir.mkdir(parents=True, exist_ok=True)

    vi_df.to_csv(out_dir / "sbm_docterm_vs_windows_vi.csv")
    nmi_df.to_csv(out_dir / "sbm_docterm_vs_windows_nmi.csv")
    ari_df.to_csv(out_dir / "sbm_docterm_vs_windows_ari.csv")

    plots.plot_clean_heatmap(
        vi_df,
        "VI: SBM DOC-TERM × SBM-Janela",
        out_dir / "sbm_docterm_vs_windows_vi.png",
        cmap="YlOrBr",
        vmax=None,
    )
    plots.plot_clean_heatmap(
        nmi_df,
        "NMI: SBM DOC-TERM × SBM-Janela",
        out_dir / "sbm_docterm_vs_windows_nmi.png",
        cmap="YlGnBu",
    )
    plots.plot_clean_heatmap(
        ari_df,
        "ARI: SBM DOC-TERM × SBM-Janela",
        out_dir / "sbm_docterm_vs_windows_ari.png",
        cmap="PuBuGn",
    )

    return vi_df, nmi_df, ari_df
