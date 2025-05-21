# Word Embedding

## Overall Project Objective

Empirically explore how the **lexical context-window size**—and other graph-construction parameters—affect document segmentation obtained with **Bayesian Stochastic Block Models (SBM)**. In addition, evaluate whether semantic embeddings produced by **Word2Vec** (CBOW/Skip-gram) and **clustered with K-Means** can **complement or explain** those structural partitions.

Current status: the work has transitioned to a comparative analysis phase, where **each combination of SBM-window and Word2Vec-window sizes** is evaluated through **cross-window heatmaps (VI, NMI, ARI)**. This extends the prior version by explicitly modeling divergences between semantic and structural segmentations.

---

## Timeline & Progress

| Phase                                                      | Period             | Key deliverables                                                                                                                                                    |
| ---------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1 – Baseline**                                           | First presentation | • Document–Term graph (spaCy + graph-tool)• Initial SBM application• Performance tweaks (dictionary lookup, `tqdm`)                                                 |
| **2 – Refinement with `main.py`**                          | Mid-project        | • Full pipeline in `main.py`• Word2Vec hyper-parameter search• Evaluation metrics (VI, NMI, Purity)• Intermediate graphs Doc–Cluster–Term and Community–Cluster     |
| **3 – Cross-window experiments (`window_experiments.py`)** | Final stage        | • Tripartite Doc–Window–Term graph• Sweep over SBM windows and Word2Vec windows independently• Matrix of VI, NMI, ARI for each pair• Heatmaps and export per metric |

---

## Theoretical Foundations & Usage Guidelines

### Motivation

SBM exploits **relational structure** derived from co-occurrence graphs built from document-window-term relationships, whereas Word2Vec captures **distributed semantics** from word co-occurrences. Each method excels in different scenarios; combining them offers the best of both worlds.

### Conceptual Comparison

**Representation & Assumptions**

* **SBM** – models link probabilities between vertices based on latent blocks, capturing structural communities.
* **Word2Vec + K-Means** – projects terms/documents into a vector space where proximity reflects semantics; K-Means assumes spherical clusters.

**Strengths**

| Approach               | Strengths                                                                                                                                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SBM**                | • Robust to textual noise when the graph is informative• Variants (degree-corrected, nested, overlapping) adapt to heterogeneity and multilayer structure |
| **Word2Vec + K-Means** | • Linear scalability with corpus size• Captures synonymy and analogies beyond purely structural models                                                    |

**Limitations**

| Approach               | Bottlenecks                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------- |
| **SBM**                | Needs a reasonably dense graph; ignores latent semantic similarity absent in topology |
| **Word2Vec + K-Means** | Requires *k* a priori; assumes convexity; loses explicit link information             |

**When to use each approach?**

* **Corpora with strong citation / collaboration structure** → **SBM** usually explains practice communities better.
* **Short corpora or lacking link metadata** → **Word2Vec** captures synonymy and local context.
* **Link-prediction tasks** → **SBM** yields probabilities for future edges.

---

## Comparative Metrics — VI, NMI & Cohesion

| Metric  | What it measures                            | Interpretation                             | Why it matters here                                                                |
| ------- | ------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------- |
| **VI**  | Information distance between two partitions | 0 = identical; higher ⇒ more divergence    | Measures dissimilarity between SBM blocks and W2V clusters                         |
| **NMI** | Shared information (0–1)                    | 1 = identical; 0 = independent             | Allows fair comparison even when block/cluster numbers differ                      |
| **ARI** | Adjusted Rand Index                         | > 0 good match; 0 random; < 0 disagreement | Captures agreement between groupings accounting for chance; confirms VI/NMI trends |

Combined reading: **Low VI + High NMI/ARI** → high overlap between structure and semantics.

---

## Execution Flow — `window_experiments.py`

| #     | Block                         | Purpose                                                      |
| ----- | ----------------------------- | ------------------------------------------------------------ |
| **0** | `main()`                      | Loads sample, NLP model, sets window list                    |
| **1** | `compare_all_partitions()`    | Main loop over all pairs (SBM-window, W2V-window)            |
| 1.1   | `initialize_graph()`          | Base graph with properties                                   |
| 1.2   | `build_window_graph()`        | Builds Doc–Window–Term graph                                 |
| 1.3   | `extract_window_term_graph()` | Extracts Window–Term for SBM                                 |
| 1.4   | `extract_doc_term_graph()`    | Aggregates Document–Term links from full graph               |
| 1.5   | `minimize_blockmodel_dl()`    | Applies SBM with constraints (type separation, edge weights) |
| 1.6   | `train_word2vec()`            | Trains Word2Vec using Skip-gram                              |
| 1.7   | `cluster_terms()`             | K-Means clustering of embeddings                             |
| 1.8   | `compare_vectors_vi_nmi()`    | Calculates VI, NMI from label vectors                        |
| 1.9   | `adjusted_rand_score()`       | Calculates ARI                                               |
| **2** | Heatmap Export                | Saves CSV + PNG for each matrix (VI, NMI, ARI)               |

**Run**

```bash
python3 window_experiments.py
```

---

## Results Summary


![VI Comparison](./outputs/window/cross_vi.png)

![NMI Comparison](./outputs/window/cross_nmi.png)

![ARI Comparison](./outputs/window/cross_ari.png)


**Adjusted Rand Index (ARI):**

* Highest values at `(5, full)` and `(5, 20)` indicate **stronger agreement** when SBM uses a narrow window and Word2Vec uses broader context.

**Normalized Mutual Information (NMI):**

* Peaks at `(5, 20)` and `(5, full)` with values of `0.59` and `0.52`, confirming those windows achieve better **semantic-structural alignment**.

**Variation of Information (VI):**

* Lowest at `(5, 20)` with `1.87`, reinforcing that this combination has the **least divergence** between groupings.

This indicates that a **narrow window for SBM** (local structure) combined with a **broad window for Word2Vec** (global semantics) yields the **highest agreement**.

---

## Conclusion — SBM vs Word2Vec + K-Means Alignment

This project aimed to evaluate how well semantic clusters generated by **Word2Vec + K-Means** align with structural communities inferred by **Stochastic Block Models (SBM)**, using **lexical windows as a modular variable**. This version extends previous work by comparing all pairwise window combinations, revealing stronger or weaker alignment patterns.

### Summary of empirical findings:

* **Best alignment** (high ARI/NMI, low VI) was achieved with **SBM window = 5** and **Word2Vec window = 20 or full**;
* Disagreements for larger SBM windows suggest **loss of local contrast**;
* **Window size asymmetry** (narrow SBM vs broad W2V) appears to work better than symmetry.

These results reinforce that **SBM and Word2Vec capture orthogonal structures**. Their disagreement is **not a flaw**, but a feature of the dual nature of structure and meaning in language.

### Implications and Future Directions

1. **Cross-window diagnostics**
   Heatmaps allow identifying windows where structural and semantic views best align, useful for parameter selection.

2. **Hybrid modeling**
   Combine embeddings as node attributes in **attribute-aware SBMs** or **multiplex networks**.

3. **Use divergence as signal**
   High VI/NMI disagreement may point to polysemy, cross-topic terms, or evolving concepts.

4. **Interactive tuning**
   User-guided interfaces could adapt window choice dynamically based on task goals.

### Final Remark

Instead of collapsing structure and semantics into a single view, this project highlights their divergence — and how it can be **measured, visualized, and used**.

