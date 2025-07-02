# Tremplin Recherche Project – Semantic and Structural Analysis of Documents

## General Objective

This project explores how different representations — **structural (SBM)** and **semantic (Word2Vec)** — can be combined to analyze large document corpora in the context of social science research. It builds heterogeneous graphs connecting documents, terms, and context windows, and compares partitions obtained via **Bayesian Stochastic Block Models (SBM)** with those obtained via **Word2Vec + KMeans**.

The project is part of the **Cortext** platform (LISIS, CNRS, INRAE, UGE), which provides advanced computational tools to researchers in the social sciences.

---

## Current Pipeline

The code now implements a **modular, multi-run experimental pipeline** with improved seed control, averaging, and visualization:

### `window_experiments.py` — Partition generation

* Selects a fixed number of documents from the corpus (`--samples`) using a user-defined `--seed`.
* For each run:
  * Builds **Document–Window–Term** graphs for multiple window sizes.
  * Applies SBM to the **Context–Window–Term** graph to extract term blocks.
  * Uses the number of SBM blocks to guide **KMeans** clustering on Word2Vec embeddings.
  * Saves **partition files** (`partitions_runXXX.parquet`) under `outputs/partitions/{samples}/seed_{seed}/model_J{window}/`.

### `compute_partition_metrics.py` — Metric computation

* Reconstructs all partition runs from saved files (grouped by seed).
* Compares:
  * **SBM × W2V**
  * **SBM × SBM**
  * **W2V × W2V**
* Computes **VI**, **NMI**, and **ARI** for each pair of windows and saves:
  * One `metrics_runXXX.parquet` per run
  * One aggregated `running_mean.parquet` per seed with full cross-window results

### `plot_average_heatmap.py` — Visualization

* Parses `running_mean.parquet`
* Generates heatmaps for each comparison type and metric:
  * `mean_nmi_sbm_w2v.png`
  * `mean_ari_sbm_sbm.png`
  * `mean_vi_w2v_w2v.png`
* Each heatmap shows **window × window** comparisons for the chosen metric

---

## Theoretical Foundation

| Method       | Description                                                      |
| ------------ | ---------------------------------------------------------------- |
| **SBM**      | Captures communities based on topological co-occurrence patterns |
| **Word2Vec** | Learns distributed representations based on lexical context      |
| **KMeans**   | Groups vectors into convex clusters                              |

The project aims to **compare and contrast** these methods to uncover divergences and complementarities between **structure** and **semantics** in language.

---

## Comparison Metrics

* **VI (Variation of Information)** – distance between partitions (lower = better)
* **NMI (Normalized Mutual Information)** – normalized similarity (higher = better)
* **ARI (Adjusted Rand Index)** – statistical concordance between partitions

All metrics are computed across all runs and averaged per window pair.

---

## Run Instructions

To execute a full experimental pipeline for a fixed sample:

1. From the root directory (`Word_Embedding/`), run:

```bash
python3 -m word_embedding.window_experiments --samples 5 --runs 5 --seed 12345
```

2. Compute metrics across all executions:

```bash
python3 word_embedding/compute_partition_metrics.py
```

3. Generate heatmaps:

```bash
python3 word_embedding/plot_average_heatmap.py
```

Each step is fully reproducible via the provided `--seed`.

---

## Results Summary

The experiments were conducted using controlled samples with fixed seeds for reproducibility.

Key takeaways include:

* **Robustness within methods**: Comparisons such as SBM × SBM and W2V × W2V across different window sizes show consistent patterns, highlighting the internal coherence of each modeling approach.

* **Semantic–structural alignment**: Cross-model comparisons (SBM × W2V) reveal cases of strong agreement (high NMI, low VI), suggesting convergence between semantic and topological partitioning in specific contexts.

* **Window impact**: Heatmaps illustrate how varying the context window affects the alignment of partitions, with some windows (e.g., 5 or full) acting as anchors of stability or divergence.

---

## Future Directions

* **Hybrid modeling**: use embeddings as node attributes in SBMs.
* **Interactive interfaces**: allow real-time visual tuning of window sizes.
* **Divergence-as-signal**: treat high disagreement as useful input for qualitative analysis.

---

Instead of forcing a single view, this project emphasizes the **productive tension** between structure and semantics — and how that tension can be **measured, visualized, and leveraged**.
