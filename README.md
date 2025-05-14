# Word Embedding

## Objetivo geral do projeto

Explorar, de forma empírica, como o **tamanho da janela de contexto** (lexical) — e outros parâmetros de construção do grafo — influenciam a segmentação de documentos obtida pelo modelo **Blocos Estocásticos Bayesianos (SBM)**, além disso, avaliar se embeddings semânticos gerados por **Word2Vec** *(CBOW/Skip-gram) e **agrupados por K-Means*** conseguem **complementar ou explicar** essas partições estruturais.

Estado atual: o trabalho concentrou‑se em prototipagem de código, execuções controladas e análise comparativa de métricas (VI, NMI, coesão), servindo como base para evoluções futuras na interface Sashimi.

---

## Linha do tempo e progresso

| Fase                                                            | Período         | Principais entregas                                                                                                                                                                                        |
| --------------------------------------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1 – Baseline**                                                | 1ª apresentação | • Grafo Documento – Termo (spaCy + Graph‑Tool)<br>• Aplicação inicial de SBM <br>• Otimizações de desempenho (lookup por dicionário, `tqdm`)                                                               |
| **2 – Refinamento com `main.py`**                               | Intermediária   | • Pipeline completo em **`main.py`** <br>• Busca de hiperparâmetros Word2Vec <br>• Métricas de avaliação (VI, NMI, Pureza) <br>• Geração de grafos intermediários Doc–Cluster–Termo e Comunidades–Clusters |
| **3 – Análise de janela de contexto (`window_experiments.py`)** | Etapa final     | • Grafo tripartido Doc–Janela–Termo <br>• Varredura de janelas 5 / 10 / 20 / 40 / 50 / FULL <br>• Comparação SBM × Word2Vec para cada janela <br>• Exportação de resultados e gráfico de coesão            |

---

## Fundamentos teóricos e diretrizes de uso

### Motivação

SBM parte da **estrutura relacional** (grafo de citações, co‑autoria, doc‑termo, etc.), enquanto Word2Vec captura **semântica distribuída** a partir das co‑ocorrências de palavras. Cada método possui pontos fortes em cenários distintos; combiná‑los oferece o melhor dos dois mundos.

### Comparação conceitual

#### Representação e premissas

* **SBM** – modela a probabilidade de ligação entre vértices com base em blocos latentes, capturando comunidades estruturais.
* **Word2Vec + K‑Means** – projeta termos/documentos em um espaço vetorial em que a proximidade reflete semântica; K‑Means assume clusters esféricos.

#### Pontos fortes

| Abordagem              | Pontos fortes                                                                                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SBM**                | • Robusta a ruído textual se o grafo for informativo <br>• Variantes (degree‑corrected, nested, overlapping) adaptam‑se a heterogeneidade e camadas múltiplas |
| **Word2Vec + K‑Means** | • Escalabilidade linear no corpus <br>• Captura sinonímia e analogias fora do alcance de modelos puramente estruturais                                        |

#### Limitações

| Abordagem              | Gargalos                                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| **SBM**                | Requer grafo relativamente denso; ignora similaridade semântica invisível na topologia    |
| **Word2Vec + K‑Means** | Precisa definir *k* a priori; assume convexidade; perde informação de ligações explícitas |

### Quando usar cada abordagem?

* **Corpus com forte estrutura de citação/colaboração** → **SBM** tende a explicar melhor as comunidades de prática.
* **Corpus curto ou sem metadados de ligação** → **Word2Vec** captura sinonímia e contexto local.
* **Tarefas de link‑prediction** → **SBM** fornece probabilidades de arestas futuras.

---

## Scripts principais

### `main.py` (Fase 2)

1. **Construção bipartida** Documento‑Termo.
2. **SBM** sobre o grafo com pesos.
3. **Word2Vec + K‑Means** com tuning opcional de hiperparâmetros.
4. **Comparações estruturais** (VI, NMI) e **Pureza** dos clusters.
5. **Visualizações**: blocos SBM, clusters, heatmaps, grafos Doc–Cluster.

### `window_experiments.py` (Fase 3)

1. **Grafo tripartido** Doc–Janela–Termo; janela = *w* tokens (ou `full`).
2. **SBM** no subgrafo Doc–Janela para obter blocos semânticos locais.
3. **Word2Vec + K‑Means** no subgrafo Doc–Termo.
4. **Métricas** por janela: número de blocos/ clusters, VI, NMI, coesão média.
5. **Relatórios** em `outputs/window` (CSVs e PDFs).

---

## Métricas comparativas — VI, NMI e Coesão

As três métricas abaixo são calculadas em paralelo para entender **como** (e **por quê**) as partições geradas por **SBM** e **Word2Vec + K-Means** convergem ou divergem.

| Métrica | O que mede | Como interpretar | Por que é útil neste projeto |
|---------|------------|------------------|------------------------------|
| **VI – Variation of Information** | Distância de informação entre duas partições | `0 = partições idênticas`;<br/>valores maiores ⇒ maior divergência | Quantifica o desvio estrutural entre comunidades do SBM e clusters semânticos; ajuda a escolher o tamanho de janela que **minimiza** discrepâncias estrutura × semântica. |
| **NMI – Normalized Mutual Information** | Informação compartilhada (normalizada de 0 a 1) | `1 = idênticas`, `0 = independentes` | Escala intuitiva; permite comparar janelas com números diferentes de blocos. Ideal para confirmar se a redução de VI reflete ganho real de similaridade. |
| **Coesão semântica**<br/>(média da similaridade Word2Vec) | Compacidade interna dos clusters | Valores altos ⇒ termos **semanticamente próximos** | Garante que um aumento de NMI/queda de VI **não** ocorre às custas de clusters difusos. Serve como controle de qualidade intracluster. |

**Leitura combinada:**  
&nbsp;&nbsp;• **VI ↓ & NMI ↑** ⇒ boa sobreposição entre SBM e Word2Vec.  
&nbsp;&nbsp;• **Coesão ↑** ⇒ essa sobreposição mantém (ou melhora) a densidade semântica interna dos grupos.

---

## 🗺️ Fluxo de Execução — `window_experiments.py`

| # | Função / Bloco | O que faz |
|---|----------------|-----------|
| **0** | **`main()`** | Define lista de janelas `[5, 10, 20, 40, 50, "full"]`, inicia contador de tempo e loop de pipelines. |
| **1** | `run_pipeline(df, nlp, win)` | Pipeline completo para cada janela **`win`**: recebe *DataFrame* (300 abstracts) e objeto **spaCy**. |
| 1.1 | `initialize_graph()` | Cria grafo vazio com propriedades (`name`, `tipo`, `color`, `amount`, etc.). |
| 1.2 | `build_window_graph(g, df, nlp, win)` | Constrói **grafo tripartido** *DOCUMENTO – JANELA – TERMO* (janelas deslizantes ou `full`). |
| 1.3 | `extract_doc_jan_graph(g)` | Deriva bipartido **DOC–JANELA** para rodar o SBM. |
| 1.4 | `extract_doc_term_graph(g)` | Deriva bipartido **DOC–TERMO** para Word2Vec + K-Means. |
| 1.5 | `min_sbm_docs_janelas(g_doc_jan)` | Aplica **SBM** (`minimize_blockmodel_dl`) ao grafo DOC–JANELA; devolve `state`. |
| 1.6 | `count_jan_blocks(g_doc_jan, state)` | Conta blocos contendo vértices-janela (base para *k* do K-Means). |
| 1.7 | `train_word2vec(df, nlp, window)` | Treina **Word2Vec** com janela contextual igual a `win`. |
| 1.8 | `cluster_terms(g_doc_term, w2v_model, n_clusters)` | Agrupa embeddings por **K-Means** (k = nº blocos). |
| 1.9 | `semantic_cohesion()` | Calcula média de similaridade intra-cluster (**Coesão**). |
| 1.10 | `cluster_analyse()` | Imprime DataFrame de rótulos, frequências e coesão; gera CSV opcional. |
| 1.11 | `compare_clusters_sbm()` | Cria comparação detalhada Cluster × Bloco SBM; exporta `cluster_sbm_w<win>_comparison.csv`. |
| 1.12 | `compare_partitions_sbm_word2vec()` | Calcula **VI**, **NMI** e matriz de overlap entre partições. |
| 1.13 | `return` → dict | Retorna métricas resumidas `{window, blocks, clusters, VI, NMI, mean_cohesion}`. |
| **2** | pós-loop | Concatena resultados em `results_window.csv` e invoca `plot_cohesion_relative_to_window()` para gerar gráficos. |
| **3** | Saída final | Mostra tabela “Resumo final” no terminal e tempo total de execução. |

### Como executar

```bash
python3 window_experiments.py
```


---

## Resultados da análise de janela 

| Janela (*w*) | Blocos SBM | Clusters W2V | VI ↓      | NMI ↑      | Coesão média ↑ |
| ------------ | ---------- | ------------ | --------- | ---------- | -------------- |
|  5           |  244       |  244         |  5.83     |  0.059     |  0.48          |
|  10          |  264       |  264         |  5.79     |  0.063     |  0.85          |
| **20**       |  **296**   |  **296**     |  **5.71** |  **0.065** |  0.90          |
|  40          |  278       |  278         |  5.76     |  0.063     |  **0.91**      |
|  50          |  267       |  267         |  5.80     |  0.061     |  0.85          |
|  FULL        |  87        |  87          |  6.24     |  0.048     |  0.88          |

### Observações

* **Janela 20 tokens** apresenta o **menor VI** (maior alinhamento estrutural) e o **maior NMI** – melhor comprometimento entre SBM e Word2Vec.
* A **coesão semântica** cresce até *w*=40 e permanece alta; contudo, o ganho estrutural não supera o custo computacional para janelas muito grandes.
* A janela **FULL** gera menos blocos e clusters, resulta em **NMI mais baixo** e maior divergência (VI), indicando que perder a granularidade local prejudica a correspondência entre métodos.


### Conclusão — janela de contexto × SBM × Word2Vec + K-Means

Os experimentos mostraram que **o tamanho da janela de contexto** afeta de maneira não linear a relação entre as comunidades estruturais inferidas pelo **SBM** e os clusters semânticos derivados de **Word2Vec + K-Means**.


**Principais insights**

1. **Janelas muito curtas** (≤ 5 tokens) geram grafos esparsos e pulverizados: o SBM forma blocos estruturais que o Word2Vec ainda não consegue explicar — VI alta, NMI baixa, coesão fraca.  
2. **Faixa intermediária (20 – 40 tokens)** apresenta o melhor compromisso:  
   * **20** tokens maximiza o alinhamento SBM × embeddings (menor VI, maior NMI).  
   * **40** tokens maximiza a qualidade interna dos clusters (maior coesão), mesmo que a semelhança estrutural caia ligeiramente.  
3. **Janelas muito largas** (≥ 50 tokens ou *FULL*) comprimem todos os termos em poucas janelas, reduzindo o número de blocos no SBM. O Word2Vec mantém alta coesão interna, mas a concordância com a estrutura do grafo despenca (VI ↑, NMI ↓).  

**Interpretação prática**

- **Balanceamento é crucial**: escolher uma janela na faixa de 20-40 tokens fornece comunidades que são **ao mesmo tempo** semanticamente compactas e estruturalmente consistentes.  
- **Coesão semântica como válvula de segurança**: garante que ganhos em NMI/VI não aconteçam à custa de “amontoar” termos semanticamente distantes.  
- Para extensões futuras (e.g. Attribute-SBM ou camadas semânticas no grafo), a janela de ~20 tokens serve de ponto de partida sólido: preserva densidade de arestas para o SBM e oferece embeddings estáveis para enriquecer o modelo.
