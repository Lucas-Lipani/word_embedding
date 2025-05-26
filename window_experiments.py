import time
import seaborn as sns
import spacy
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Usa backend para salvar arquivos, sem abrir janelas
from graph_tool.all import (
    Graph,
    prop_to_size,
    LayeredBlockState,
    graph_draw,
    sfdp_layout,
    minimize_blockmodel_dl,
    variation_information,
    mutual_information,
    partition_overlap,
)
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def initialize_graph():
    """
    Inicializa e configura um grafo não direcionado com as propriedades básicas baseado no projeto Sashimi.

    :return: Um objeto Graph vazio com propriedades de vértice e de aresta definidas.
    """
    g = Graph(directed=False)

    name_prop = g.new_vertex_property("string")
    tipo_prop = g.new_vertex_property("int")
    short_term_prop = g.new_vertex_property("string")
    color_prop = g.new_vertex_property("vector<double>")
    posicao_prop = g.new_vertex_property("vector<double>")
    amount_prop = g.vp["amount"] = g.new_vertex_property("int")
    size_prop = g.new_vertex_property("double")

    weight_prop = g.new_edge_property("long")
    layer_prop = g.new_edge_property("int")

    g.vp["amount"] = amount_prop
    g.ep["layer"] = layer_prop
    g.vp["size"] = size_prop
    g.vp["color"] = color_prop
    g.vp["name"] = name_prop
    g.vp["tipo"] = tipo_prop
    g.vp["short_term"] = short_term_prop
    g.vp["posicao"] = posicao_prop
    g.ep["weight"] = weight_prop

    return g


def draw_base_graphs(g, g_doc_jan, g_doc_term, window):

    window = str(window)

    # Salva o grafo original DOCUMENTO - JANELAS - TERMOS
    graph_draw(
        g,
        # pos=sfdp_layout(g),           # Layout para posicionar os nós
        pos=g.vp["posicao"],
        vertex_text=g.vp["name"],  # Usa o rótulo armazenado na propriedade "name"
        vertex_text_position=-2,
        vertex_text_color="black",
        vertex_font_size=10,  # Tamanho da fonte dos rótulos
        vertex_fill_color=g.vp["color"],  # Define a cor dos vértices
        output="outputs/window/window"
        + window
        + "_graph_d-j-t.pdf",  # Salva a visualização em PDF
    )

    # Salva o grafo original DOCUMENTO - JANELAS
    graph_draw(
        g_doc_jan,
        # pos=sfdp_layout(g_doc_jan),           # Layout para posicionar os nós
        pos=g_doc_jan.vp["posicao"],
        vertex_text=g_doc_jan.vp[
            "name"
        ],  # Usa o rótulo armazenado na propriedade "name"
        vertex_text_position=-2,
        vertex_text_color="black",
        vertex_font_size=10,  # Tamanho da fonte dos rótulos
        vertex_fill_color=g_doc_jan.vp["color"],  # Define a cor dos vértices
        output="outputs/window/window"
        + window
        + "_graph_d-j.pdf",  # Salva a visualização em PDF
    )

    # Salva o grafo original DOCUMENTO - TERMOS
    graph_draw(
        g_doc_term,
        # pos=sfdp_layout(g_doc_term),           # Layout para posicionar os nós
        pos=g_doc_term.vp["posicao"],
        vertex_text=g_doc_term.vp[
            "name"
        ],  # Usa o rótulo armazenado na propriedade "name"
        vertex_text_position=-2,
        vertex_text_color="black",
        vertex_font_size=10,  # Tamanho da fonte dos rótulos
        vertex_fill_color=g_doc_term.vp["color"],  # Define a cor dos vértices
        output="outputs/window/window"
        + window
        + "_graph_d-t.pdf",  # Salva a visualização em PDF
    )


def build_window_graph(g, df, nlp, w):
    """
    Constrói o grafo tripartido Documento – Janela de contexto – Termos num formato
    multiplexo de 2 camadas:

      camada 0 : arestas Janela de contexto → Termo central
      camada 1 : arestas Janela de contexto → cada termo de contexto que compõe a janela
    """
    g.vp["termos"] = g.new_vertex_property("object")

    window_vertex = {}  # chave (frozenset(win_tokens), term_central) → v_jan
    term_vertex = {}  # termo → v_term
    doc_vertex = {}  # doc_id → v_doc
    doc_y = term_y = win_y = 0  # coordenadas para layout

    for idx, row in tqdm(
        df.iterrows(), desc="Processando Doc-Jan-Termos", total=len(df)
    ):
        doc_id = str(idx)
        abstract = row["abstract"]

        # ───── vértice Documento ─────
        v_doc = g.add_vertex()
        g.vp["name"][v_doc], g.vp["tipo"][v_doc] = doc_id, 0
        g.vp["posicao"][v_doc] = [-15, doc_y]
        doc_y += 1
        g.vp["size"][v_doc], g.vp["color"][v_doc] = 20, [1, 0, 0, 1]
        doc_vertex[doc_id] = v_doc

        # tokenização
        toks = [
            t.text.lower().strip()
            for t in nlp(abstract)
            if not t.is_stop and not t.is_punct
        ]

        #  tamanho da janela de contexto
        if w == "full":
            w_local = len(toks)
        else:
            w_local = int(w)

        # ───── janelas centradas em cada token ─────
        for i, term_central in enumerate(toks):
            start, end = max(0, i - w_local), min(len(toks), i + w_local + 1)
            win_tokens = toks[start:i] + toks[i + 1 : end]  # contexto
            win_key = (frozenset(win_tokens), term_central)  # deduplicação

            # --- vértice Janela ---
            if win_key not in window_vertex:
                v_win = g.add_vertex()
                g.vp["name"][v_win] = " ".join(win_tokens)  
                g.vp["tipo"][v_win] = 3
                g.vp["termos"][v_win] = win_tokens
                g.vp["posicao"][v_win] = [0, win_y]
                win_y += 2
                g.vp["size"][v_win] = 15
                g.vp["color"][v_win] = [0.6, 0.6, 0.6, 1]
                window_vertex[win_key] = v_win
            else:
                v_win = window_vertex[win_key]

            # ligar Doc → Janela
            if g.edge(doc_vertex[doc_id], v_win) is None:
                g.add_edge(doc_vertex[doc_id], v_win)

            # --- vértice Termo Central ---
            if term_central not in term_vertex:
                v_term = g.add_vertex()
                g.vp["name"][v_term] = term_central
                g.vp["tipo"][v_term] = 1
                g.vp["posicao"][v_term] = [15, term_y]
                term_y += 1
                g.vp["size"][v_term] = 10
                g.vp["color"][v_term] = [0, 0, 1, 1]
                g.vp["amount"][v_term] = 1
                term_vertex[term_central] = v_term
            else:
                v_term = term_vertex[term_central]
                g.vp["amount"][v_term] += 1

            # ╭─────────── camada 0: Jan → Termo central ───────────╮
            e0 = g.edge(v_win, v_term)
            if e0 is None:
                e0 = g.add_edge(v_win, v_term)
                g.ep["weight"][e0] = 1
                g.ep["layer"][e0] = 0
            else:
                g.ep["weight"][e0] += 1
            # ╰──────────────────────────────────────────────────────╯

            # ╭─────────── camada 1: Jan → cada termo de contexto ───╮
            for tok in win_tokens:
                if tok not in term_vertex:
                    v_tok = g.add_vertex()
                    g.vp["name"][v_tok] = tok
                    g.vp["tipo"][v_tok] = 1
                    g.vp["posicao"][v_tok] = [15, term_y]
                    term_y += 1
                    g.vp["size"][v_tok] = 10
                    g.vp["color"][v_tok] = [0, 0, 1, 1]
                    g.vp["amount"][v_tok] = 1
                    term_vertex[tok] = v_tok
                else:
                    v_tok = term_vertex[tok]
                    g.vp["amount"][v_tok] += 1

                e1 = g.edge(v_win, v_tok)
                if e1 is None:
                    e1 = g.add_edge(v_win, v_tok)
                    g.ep["weight"][e1] = 1
                    g.ep["layer"][e1] = 1
                else:
                    g.ep["weight"][e1] += 1
            # ╰──────────────────────────────────────────────────────╯

    return g


def extract_window_term_graph(g):
    """
    Extrai um subgrafo contendo apenas vértices de tipo JANELA (3) e TERMO (1),
    e as arestas que os conectam no grafo original.

    :param g: Grafo completo DOCUMENTO–JANELA–TERMO
    :return: Subgrafo JANELA–TERMO
    """
    g_win_term = Graph(directed=False)

    # Copiar propriedades
    for prop in g.vp.keys():
        g_win_term.vp[prop] = g_win_term.new_vertex_property(g.vp[prop].value_type())
    for prop in g.ep.keys():
        g_win_term.ep[prop] = g_win_term.new_edge_property(g.ep[prop].value_type())

    # Mapear vértices válidos
    vertex_map = {}
    for v in g.vertices():
        if g.vp["tipo"][v] in (1, 3):  # TERMO ou JANELA
            v_new = g_win_term.add_vertex()
            vertex_map[int(v)] = v_new
            for prop in g.vp.keys():
                g_win_term.vp[prop][v_new] = g.vp[prop][v]

    # Adicionar arestas válidas
    for e in g.edges():
        src = int(e.source())
        tgt = int(e.target())
        if src in vertex_map and tgt in vertex_map:
            e_new = g_win_term.add_edge(vertex_map[src], vertex_map[tgt])
            for prop in g.ep.keys():
                g_win_term.ep[prop][e_new] = g.ep[prop][e]

    return g_win_term


def extract_doc_term_graph(g):
    """
    Cria um grafo bipartido Documento–Termo com pesos agregados,
    baseando-se nas conexões Documento → Janela → Termo.

    Para cada doc–termo, soma quantas vezes o termo aparece nas janelas do documento.
    """
    g_doc_term = Graph(directed=False)

    # Criar propriedades
    for prop in g.vp.keys():
        g_doc_term.vp[prop] = g_doc_term.new_vertex_property(g.vp[prop].value_type())
    for prop in g.ep.keys():
        g_doc_term.ep[prop] = g_doc_term.new_edge_property(g.ep[prop].value_type())

    vertex_map = {}
    term_ids = {}
    doc_ids = {}

    # Copiar documentos e termos
    for v in g.vertices():
        tipo = g.vp["tipo"][v]
        if tipo == 0 or tipo == 1:
            v_new = g_doc_term.add_vertex()
            vertex_map[int(v)] = v_new
            for prop in g.vp.keys():
                g_doc_term.vp[prop][v_new] = g.vp[prop][v]

            if tipo == 0:
                doc_ids[int(v)] = v_new
            else:
                term_ids[int(v)] = v_new

    # Acumular pesos de doc → termo via janelas
    for v_doc in g.vertices():
        if g.vp["tipo"][v_doc] != 0:
            continue
        doc_id = int(v_doc)
        term_weights = Counter()

        # para cada janela conectada ao doc
        for e_doc_win in v_doc.out_edges():
            v_win = e_doc_win.target()
            if g.vp["tipo"][v_win] != 3:
                continue

            # para cada termo conectado à janela
            for e_win_term in v_win.out_edges():
                v_term = e_win_term.target()
                if g.vp["tipo"][v_term] != 1:
                    continue
                term_id = int(v_term)
                weight = g.ep["weight"][e_win_term]
                term_weights[term_id] += weight

        # criar arestas no grafo doc-termo
        for term_id, weight in term_weights.items():
            v_doc_new = doc_ids[doc_id]
            v_term_new = term_ids[term_id]
            e = g_doc_term.add_edge(v_doc_new, v_term_new)
            g_doc_term.ep["weight"][e] = weight

    return g_doc_term


def cluster_terms(g, w2v_model, n_clusters):
    """
    Realiza a clusterização dos termos do grafo utilizando os vetores semânticos do modelo Word2Vec.

    :param g: Grafo bipartido com a relação DOCUMENTOS - TERMOS.
    :param w2v_model: Modelo Word2Vec previamente treinado, usado para extrair vetores semânticos de termos.
    :param n_clusters: Número de clusters a serem formados, geralmente definido a partir do número de comunidades identificadas pelo SBM.
    :return: Dicionário onde as chaves são os rótulos dos clusters e os valores são listas de vértices (termos) que pertencem a cada cluster.
    """
    cluster_prop = g.new_vertex_property("int")
    g.vp["cluster"] = cluster_prop

    term_indices = []
    term_vectors = []
    # Percorrer os vértices do grafo e buscar os vetores significativos de termos.
    for v in g.vertices():
        if int(g.vp["tipo"][v]) == 1:
            term = g.vp["name"][v]
            try:
                vec = w2v_model.wv[term]
                if np.linalg.norm(vec) > 0:
                    term_indices.append(int(v))
                    term_vectors.append(vec)
            except KeyError:
                pass

    if len(term_vectors) == 0:
        print("Nenhum vetor de termo foi encontrado.")
        return {}

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(term_vectors)

    idx = 0
    clusters = {}
    for v in g.vertices():
        if int(g.vp["tipo"][v]) == 1:
            if idx >= len(labels):
                continue
            label = int(labels[idx])
            g.vp["cluster"][v] = label
            clusters.setdefault(label, []).append(v)
            idx += 1

    return clusters


def train_word2vec(df, nlp, window):
    """
    Processa os abstracts do corpus e treina um modelo Word2Vec usando os tokens obtidos.

    :param df: DataFrame contendo uma coluna "abstract" com os textos a serem processados.
    :param nlp: Tokenizer do spaCy (por exemplo, "en_core_web_sm") para processar os textos e remover stop words e pontuações.
    :param window: Tamanho da janela de conteto para o treinamento do modelo.
    :return: Um modelo Word2Vec treinado com os tokens extraídos dos abstracts.
    """
    sentences = []
    for abstract in tqdm(df["abstract"], desc="Pré-processamento para Word2Vec"):
        doc = nlp(abstract)
        tokens = [
            token.text.lower().strip()
            for token in doc
            if not token.is_stop and not token.is_punct
        ]
        sentences.append(tokens)

    model = Word2Vec(
        sentences,  # Lista de abstracts tokenizados usados para treinar o modelo
        vector_size=100,  # Tamanho do vetor de representação para cada palavra (100 dimensões)
        window=window,  # Número de palavras antes e depois da palavra-alvo consideradas no contexto (janela de contexto)
        min_count=1,  # Ignora palavras que aparecem menos de 2 vezes no corpus
        sg=1,  # 1 para skip-gram ou 0 (default) para CBOW. CBOW: contexto ➜ palavra | Skip‑gram: palavra ➜ contexto
        workers=4,  # Número de threads utilizadas para acelerar o treinamento
        epochs=15,
    )

    return model


def count_connected_term_blocks(state, g):
    """Retorna quantidade de blocos de termo que possuem um ou mais arestas.
    """
    blocks_vec = state.get_blocks().a
    block_graph = state.get_bg()  # para LayeredBlockState

    # blocos cujo grau total > 0
    connected = {
        int(v)
        for v in block_graph[0].vertices()
        if (v.out_degree() + v.in_degree()) > 0
    }
    term_blocks = set()
    for v in g.vertices():
        if int(g.vp["tipo"][v]) == 1:
            b = int(blocks_vec[int(v)])
            if b in connected:
                term_blocks.add(b)
    return len(term_blocks)


def compare_vectors_vi_nmi(sbm_labels, w2v_labels):
    """Calcula VI, MI, NMI para dois vetores de rótulos numpy."""
    vi = variation_information(sbm_labels, w2v_labels)
    mi = mutual_information(sbm_labels, w2v_labels)
    po = partition_overlap(sbm_labels, w2v_labels)
    nmi = po[2] if isinstance(po, (tuple, list, np.ndarray)) else po
    return vi, mi, nmi


def compare_all_partitions(df, nlp, window_list):
    """Para cada par (w_sbm, w_w2v) calcula VI / NMI / ARI e devolve três matrizes."""

    # dataframes de saída
    results_vi = pd.DataFrame(index=window_list, columns=window_list)
    results_nmi = pd.DataFrame(index=window_list, columns=window_list)
    results_ari = pd.DataFrame(index=window_list, columns=window_list)

    # cache de modelos Word2Vec: { janela : modelo }
    w2v_models = {}

    # ----------------------------------------------------
    # 1. loop sobre janela do SBM (w_sbm)
    # ----------------------------------------------------
    for w_sbm in window_list:
        print(f"\n### SBM janela = {w_sbm}")

        # (a) construir grafo completo + SBM em JAN-TERM
        g_full = initialize_graph()
        g_full = build_window_graph(g_full, df, nlp, w_sbm)
        print("Grafo DOC-JAN-TERM:")
        print(g_full)

        g_jan_term = extract_window_term_graph(g_full)
        print("Grafo JAN-TERM:")
        print(g_jan_term)

        doc_term = extract_doc_term_graph(g_full)
        print("Grafo DOC-TERM:")
        print(doc_term)

        # # #  # Impressão dos 3 grafos bases do projeto
        # draw_base_graphs(g_full,g_jan_term,doc_term, w_sbm)
        # exit()

        state = minimize_blockmodel_dl(
            g_jan_term,
            state=LayeredBlockState,  # modelo adequado a camadas
            state_args=dict(
                ec=g_jan_term.ep["layer"],  # mapa de camadas/discretas
                layers=True,  # versão “camadas independentes”
                eweight=g_jan_term.ep["weight"],  # (opcional) multiplicidade da aresta
                pclabel=g_jan_term.vp[
                    "tipo"
                ],  # mantém janelas e termos em grupos separados
            ),
        )
        # Refinar com MCMC
        # state.mcmc_sweep(niter=1000)  # Tenta mudar rótulos de vértices localmente

        print("A saída do SBM:")
        print(state)
        # state = state.project_level(0)   # nível mais detalhado, use a função do nested e queira trabalhar como se não fosse nested.

        # definição da quantidade de clusters através do números de bocos de termos
        k_blocks = count_connected_term_blocks(state, g_jan_term)
        print(f"   blocos SBM (com termos e conexões) = {k_blocks}")

        # mapa termo → bloco (preciso para vetores paralelos)
        blocks_vec = state.get_blocks().a
        term_to_block = {
            g_jan_term.vp["name"][v]: int(blocks_vec[int(v)])
            for v in g_jan_term.vertices()
            if int(g_jan_term.vp["tipo"][v]) == 1
        }

        # ------------------------------------------------
        # 2. loop sobre janela Word2Vec (w_w2v)
        # ------------------------------------------------
        for w_w2v in window_list:
            print(f"      → Word2Vec janela = {w_w2v}")

            # treino Word2Vec apenas 1ª vez
            if w_w2v not in w2v_models:
                w_int = 10000 if w_w2v == "full" else w_w2v
                w2v_models[w_w2v] = train_word2vec(df, nlp, w_int)
            w2v_model = w2v_models[w_w2v]

            # clonar grafo Doc‑Term para clusterização isolada
            g_dt = doc_term.copy()
            clusters = cluster_terms(g_dt, w2v_model, n_clusters=k_blocks)

            # montar vetores paralelos
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

            if len(set(w2v_labels)) > 1 and len(set(sbm_labels)) > 1:
                sbm_arr = np.array(sbm_labels)
                w2v_arr = np.array(w2v_labels)
                vi, mi, nmi = compare_vectors_vi_nmi(sbm_arr, w2v_arr)
                ari = adjusted_rand_score(sbm_arr, w2v_arr)
            else:
                vi = nmi = ari = np.nan

            results_vi.loc[w_sbm, w_w2v] = vi
            results_nmi.loc[w_sbm, w_w2v] = nmi
            results_ari.loc[w_sbm, w_w2v] = ari

    # salvar resultados
    results_vi.to_csv("outputs/window/matriz_vi.csv")
    results_nmi.to_csv("outputs/window/matriz_nmi.csv")
    results_ari.to_csv("outputs/window/matriz_ari.csv")

    return results_vi, results_nmi, results_ari


def plot_clean_heatmap(matrix, title, filename, cmap, vmin=0, vmax=1):
    matrix_plot = matrix.astype(float).fillna(-1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix_plot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
    )

    for text in ax.texts:
        if text.get_text() == "-1.00":
            text.set_text("N/A")

    ax.set_title(title)
    ax.set_xlabel("Janela Word2Vec")
    ax.set_ylabel("Janela SBM")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    start = time.time()

    nlp = spacy.load("en_core_web_sm")
    df = pd.read_parquet("wos_sts_journals.parquet").sample(n=300, random_state=42)

    WINDOW_LIST = [5, 10, 20, 40, 50, "full"]

    vi_mat, nmi_mat, ari_mat = compare_all_partitions(df, nlp, WINDOW_LIST)

    # plot heatmaps
    plot_clean_heatmap(
        nmi_mat, "NMI: SBM x Word2Vec", "outputs/window/cross_nmi.png", cmap="YlGnBu"
    )
    plot_clean_heatmap(
        vi_mat,
        "VI: SBM x Word2Vec",
        "outputs/window/cross_vi.png",
        cmap="YlOrBr",
        vmax=None,
    )
    plot_clean_heatmap(
        ari_mat, "ARI: SBM x Word2Vec", "outputs/window/cross_ari.png", cmap="PuBuGn"
    )

    print(f"\nTempo total: {time.time() - start:.2f} s")
