# Projeto: Análise de Comunidades e Semântica

Este repositório contém o código e os experimentos desenvolvidos no âmbito do **Tremplin Recherche** para analisar e agrupar documentos (ou termos) usando duas abordagens complementares — **Stochastic Block Model (SBM)** e **Word2Vec + K‑Means** — além de propor extensões híbridas e melhorias individuais.

---

## Índice

1. [Motivação](#motivação)
2. [Comparação conceitual](#comparação-conceitual)
3. [Quando usar cada abordagem?](#quando-usar-cada-abordagem)
4. [Ideias para um modelo híbrido](#ideias-para-um-modelo-híbrido)
5. [Melhorias individuais](#melhorias-individuais)
6. [Como integrar no código](#como-integrar-no-código)
7. [Próximos experimentos](#próximos-experimentos)

---

## Motivação

SBM parte da **estrutura relacional** (grafo de citações, co‑autoria, doc‑termo, etc.) enquanto Word2Vec captura **semântica distribuída** a partir de co‑ocorrências de palavras. Cada método tem pontos fortes em cenários distintos; combiná‑los pode oferecer o melhor dos dois mundos.

---

## Comparação conceitual

### Representação e premissas

* **SBM** modela a probabilidade de ligação entre vértices com base em blocos latentes, capturando comunidades estruturais.
* **Word2Vec + K‑Means** projeta termos/documentos num espaço vetorial onde a proximidade reflete semântica; K‑Means assume clusters esféricos.

### Pontos fortes

| Abordagem              | Pontos fortes                                                                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **SBM**                | • Robusto a ruído textual se o grafo for informativo<br>• Variantes (degree‑corrected, nested, overlapping) adaptam‑se a heterogeneidade e multilayers |
| **Word2Vec + K‑Means** | • Escalabilidade linear no corpus<br>• Captura sinonímia e analogias fora do alcance de modelos puramente estruturais                                  |

### Limitações

| Abordagem              | Gargalos                                                                           |
| ---------------------- | ---------------------------------------------------------------------------------- |
| **SBM**                | Requer grafo denso; ignora similaridade semântica invisível na topologia           |
| **Word2Vec + K‑Means** | Necessita definir *k*, assume convexidade; perde informação de ligações explícitas |

---

## Quando usar cada abordagem?

* **Corpus com forte estrutura de citação/colaboração** → **SBM** tende a explicar melhor comunidades de prática.
* **Corpus curto ou sem metadados de ligação** → **Word2Vec** captura sinonímia e contexto local.
* **Tarefas de link‑prediction ou recomendação** → **SBM** fornece probabilidades de arestas futuras.

---

## Ideias para um modelo híbrido

| Estratégia                       | Descrição                                                                     | Dicas de implementação                                                                      |
| -------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Ensemble 2‑etapas**            | 1️⃣ Rodar SBM para blocos macro → 2️⃣ Word2Vec + K‑Means dentro de cada bloco | No `main.py`, após inferir blocos, iterar sobre vértices do bloco e aplicar `cluster_terms` |
| **Attribute‑SBM**                | Embeddings como covariáveis; combina adjacência **+** distância vetorial      | Criar arestas semânticas (cosine > τ) e usar `minimize_blockmodel_dl` em modo multilayer    |
| **GNN + regularizador SBM**      | GNN gera embeddings e é penalizada pela entropia do SBM                       | Integrar PyTorch com graph‑tool, retro‑propagando a NLL do SBM                              |
| **Fuse‑then‑cluster (spectral)** | Concatena vetores Word2Vec + autovetores da modularidade antes de K‑Means     | `np.hstack([spectral_embs, term_vectors])` → `KMeans`                                       |

---

## Melhorias individuais

### SBM

1. **Degree‑corrected SBM** para evitar que hubs dominem blocos.
2. **Nested SBM** para hierarquia automática.
3. **Priors informativos**: usar metadados (ano, revista) como rótulos parciais.
4. **Seleção bayesiana de modelo**: comparar evidência marginal entre variantes.

### Word2Vec + K‑Means

1. **Detecção de *phrases*** (bigrams/trigrams) antes do treino.
2. **Spherical K‑Means ou HDBSCAN** para clusters de tamanhos díspares.
3. **Métricas de qualidade**: *silhouette*, *topic coherence*.
4. **Fine‑tuning supervisionado leve** com amostras positivas/negativas (Gensim).
5. **Doc2Vec ou SBERT** como alternativa a somatório de vetores.

---

## Como integrar no código

### Exemplo: criar camada semântica antes do SBM

```python
from itertools import combinations
from scipy.spatial.distance import cosine

for v_i, v_j in combinations(term_vertices, 2):
    sim = 1 - cosine(w2v[v_i], w2v[v_j])
    if sim > 0.6:
        e = g.add_edge(v_i, v_j)
        g.ep['weight'][e] = sim  # nova property
```

Em seguida:

```python
state = gt.minimize_blockmodel_dl(g, state_args={"layers": 2})
```

### Attribute‑SBM (graph‑tool ≥ 2.61)

```python
init_b = kmeans_init(term_vectors, k)
state = gt.BlockState(g, b=init_b, recs=[term_vectors], rec_types=["real-normal"])
state.multiflip_mcmc_sweep(niter=1000)
```

---

## Próximos experimentos

1. **Grid de *k* adaptativo**: usar *silhouette* em vez de fixar `num_blocos_termo`.
2. **Análise temporal**: janelas deslizantes por ano para checar estabilidade de blocos.
3. **Human‑in‑the‑loop**: registrar feedback de especialistas e usar como rótulos parciais para refinamento.
