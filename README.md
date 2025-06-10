# Tremplin Recherche Project – Semantic and Structural Analysis of Documents

## General Objective

This project explores how different representations — **structural (SBM)** and **semantic (Word2Vec)** — can be combined to analyze large document corpora in the context of social science research. It builds heterogeneous graphs connecting documents, terms, and context windows, and compares partitions obtained via **Bayesian Stochastic Block Models (SBM)** with those obtained via **Word2Vec + KMeans**.

The project is part of the **Cortext** platform (LISIS, CNRS, INRAE, UGE), which provides advanced computational tools to researchers in the social sciences.

---

## Current Pipeline

The code implements a **cross-window experimental pipeline**:

### `window_experiments.py` — Cross-window comparison

* Builds **Document–Window–Term graphs** for multiple window sizes.
* Extracts relevant subgraphs: Window–Term, Document–Term, Context–Window–Term.
* Applies SBM over the **Context–Window–Term** graph to get term blocks.
* Uses the number of SBM blocks as `k` for **KMeans**.
* Compares **SBM × Word2Vec partitions** for all combinations of window sizes:

  * Outputs **VI**, **NMI**, **ARI** matrices.
  * Plots **heatmaps** for visual inspection.

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

Additional measures:

* **Semantic cohesion** – average pairwise similarity within Word2Vec clusters
* **Cluster purity** – proportion of cluster terms belonging to the dominant SBM block

---


## Run Instructions

To run the experiment pipeline properly:

1. **Ensure you are in the `src/` directory.**
2. **Use Python’s module execution:**

```bash
cd src
python3 -m word_embedding

```
Results are saved under `outputs/window/`.

---

## Results Summary

The experiments were conducted using a **subset of 300 documents** extracted from a corpus of over 20,000 entries. This reduced sample limits granularity and statistical representativeness but provides a clear proof of concept.

Key takeaways include:

* **Self-consistency across models**: NMI comparisons within SBM (SBM × SBM) and within Word2Vec (W2V × W2V) showed stable patterns across most window sizes, suggesting robustness in partitioning methods despite data sparsity.

* **Cross-method comparison**: When comparing SBM and Word2Vec partitions using identical or varying window sizes, the results revealed both alignment and divergence zones. Notably, specific combinations such as SBM window = 5 and Word2Vec window = 20 showed promising NMI and low VI values, indicating potential convergence in how structure and semantics organize terms.

* **SBM vs SBM-DOC-TERM**: Direct comparisons between context-sensitive SBM blocks and a baseline SBM over the DOC–TERM graph showed moderate alignment, supporting the idea that local context windows introduce meaningful variation in structural partitioning.

Overall, these results illustrate that even with a limited sample size, the methodology is capable of highlighting consistent structural-semantic relationships — which are expected to become more precise and expressive when applied to the full corpus.

---

## Future Directions

* **Hybrid modeling**: use embeddings as node attributes in SBMs.
* **Interactive interfaces**: allow real-time visual tuning of window sizes.
* **Divergence-as-signal**: treat high disagreement as useful input for qualitative analysis.

---

Instead of forcing a single view, this project emphasizes the **productive tension** between structure and semantics — and how that tension can be **measured, visualized, and leveraged**.
