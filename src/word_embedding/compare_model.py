import pandas as pd
from pathlib import Path
import numpy as np
import plots
from graph_tool.all import (variation_information, mutual_information, partition_overlap)
from sklearn.metrics import adjusted_rand_score


def compare_same_model_partitions(model_outputs, window_list, model_name="SBM"):
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