import time
from itertools import product
import spacy
from gensim.models import Word2Vec
from copy import deepcopy
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Usa backend para salvar arquivos, sem abrir janelas
import seaborn as sns
from heapq import nlargest
from graph_tool.all import (Graph, prop_to_size, graph_draw, sfdp_layout, minimize_blockmodel_dl, variation_information, mutual_information, partition_overlap)
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity



def initialize_graph():
    """
    Inicializa e configura um grafo não direcionado com as propriedades básicas para o projeto Sashimi.
    
    :return: Um objeto Graph com propriedades de vértice (name, tipo, short_term, color, posicao, amount, size) e de aresta (weight) definidas.
    """
    g = Graph(directed=False)

    name_prop = g.new_vertex_property("string")
    tipo_prop = g.new_vertex_property("int")
    short_term_prop = g.new_vertex_property("string")
    color_prop = g.new_vertex_property("vector<double>")
    posicao_prop = g.new_vertex_property("vector<double>")
    amount_prop = g.vp["amount"] = g.new_vertex_property("int")
    size_prop = g.new_vertex_property("double")
    weight_prop = g.new_edge_property("int")

    g.vp["amount"] = amount_prop
    g.vp["size"] = size_prop
    g.vp["color"] = color_prop
    g.vp["name"] = name_prop
    g.vp["tipo"] = tipo_prop
    g.vp["short_term"] = short_term_prop
    g.vp["posicao"] = posicao_prop
    g.ep["weight"] = weight_prop
    
    return g

def run_word2vec_param_search(df, nlp, g_base, state_sbm, param_list, n_clusters):

    results = []

    for params in param_list:
        print(f"Testing Word2Vec params: {params}")

        # Preprocess abstracts into token lists
        sentences = []
        for abstract in df["abstract"]:
            doc = nlp(abstract)
            tokens = [
                token.text.lower().strip()
                for token in doc
                if not token.is_stop and not token.is_punct
            ]
            sentences.append(tokens)

        # Train Word2Vec model with current params
        model = Word2Vec(
            sentences=sentences,
            vector_size=params["vector_size"],
            window=params["window"],
            min_count=params["min_count"],
            sg=params["sg"],
            workers=4
        )

        # Copy base graph
        g = deepcopy(g_base)

        # Cluster terms using current model
        clusters = cluster_terms(g, model, n_clusters=n_clusters)
        if not clusters:
            print("Skipping configuration due to no clusters.")
            continue

        # Compute cohesion
        cohesion_scores = semantic_cohesion(g, clusters, model)

        # Compare partitions (SBM vs Word2Vec)
        try:
            vi, mi, po = compare_partitions_sbm_word2vec(g, state_sbm)
            nmi = po if isinstance(po, float) else po[2] if len(po) > 2 else 0.0
        except Exception as e:
            print(f"Error comparing partitions: {e}")
            continue

        # Compute cluster purity
        df_pureza = compute_cluster_purity(clusters, state_sbm, g)
        mean_purity = df_pureza["Pureza"].mean() if not df_pureza.empty else 0.0

        results.append({
            "params": params,
            "VI": vi,
            "MI": mi,
            "NMI": nmi,
            "mean_purity": mean_purity
        })

    df_results = pd.DataFrame(results)
    df_results["vector_size"] = df_results["params"].apply(lambda x: x["vector_size"])
    df_results["window"] = df_results["params"].apply(lambda x: x["window"])
    df_results["min_count"] = df_results["params"].apply(lambda x: x["min_count"])
    df_results["sg"] = df_results["params"].apply(lambda x: x["sg"])

    df_results.to_csv("outputs/main/resultados_parametros_word2vec.csv", index=False)
    print("\nResumo dos melhores parâmetros:\n", df_results.sort_values("NMI", ascending=False).head())

    # Escolher melhor configuração (maior NMI)
    best_row = df_results.sort_values("NMI", ascending=False).iloc[0]

    best_params = {
        "vector_size": int(best_row["vector_size"]),
        "window": int(best_row["window"]),
        "min_count": int(best_row["min_count"]),
        "sg": int(best_row["sg"])
    }

    print("Melhores hiperparâmetros selecionados:", best_params)

    # ── Re‑treinar o Word2Vec com os melhores hiperparâmetros ──
    sentences = []
    for abstract in df["abstract"]:
        doc = nlp(abstract)
        tokens = [
            token.text.lower().strip()
            for token in doc
            if not token.is_stop and not token.is_punct
        ]
        sentences.append(tokens)

    best_model = Word2Vec(
        sentences=sentences,
        vector_size=best_params["vector_size"],
        window=best_params["window"],
        min_count=best_params["min_count"],
        sg=best_params["sg"],
        workers=4
    )

    # (opcional) salvar CSV já sem a coluna params
    df_results.drop(columns=["params"], inplace=True)
    df_results.to_csv("outputs/main/resultados_parametros_word2vec.csv", index=False)

    return best_model, best_params["window"]


def teste_hiperparametros(df, nlp, param_list, param_grid):
    g_base = initialize_graph()
    g_base = build_bipartite_graph(g_base, df, nlp)

    # Aplicar SBM
    state_wew = min_sbm_wew(g_base)
    num_blocos_termo = count_term_blocks(g_base, state_wew)

    w2v_model, window = run_word2vec_param_search(df, nlp, g_base, state_wew, param_list, num_blocos_termo)

    return w2v_model, window

def train_word2vec(df, nlp):
    """
    Processa os abstracts do corpus e treina um modelo Word2Vec usando os tokens obtidos.
    
    :param df: DataFrame contendo uma coluna "abstract" com os textos a serem processados.
    :param nlp: Tokenizer do spaCy (por exemplo, "en_core_web_sm") para processar os textos e remover stop words e pontuações.
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
        sentences,      # Lista de abstracts tokenizados usados para treinar o modelo
        vector_size=100,  # Tamanho do vetor de representação para cada palavra (100 dimensões)
        window=10,       # Número de palavras antes e depois da palavra-alvo consideradas no contexto (janela de contexto)
        min_count=2,    # Ignora palavras que aparecem menos de 2 vezes no corpus
        sg=0,           # 1 para skip-gram ou 0 (default) para CBOW. CBOW: contexto ➜ palavra | Skip‑gram: palavra ➜ contexto
        workers=4       # Número de threads utilizadas para acelerar o treinamento
    )
 
    return model, model.window


def build_bipartite_graph(g, df, nlp):
    """
    Constrói um grafo bipartido relacionando documentos e termos a partir dos abstracts do corpus.
    
    :param g: Objeto Graph vazio com propriedades pré-definidas, onde serão inseridos os vértices de documentos e termos.
    :param df: DataFrame contendo os abstracts dos documentos que serão processados.
    :param nlp: Tokenizer do spaCy para processar os abstracts, removendo stop words e pontuações.
    :return: Grafo bipartido com vértices representando DOCUMENTOS e TERMOS, e arestas ponderadas pela frequência dos termos.
    """
    doc_vertex = {}
    term_vertex = {}
    doc_y = 0
    term_y = 0
    for index, row in tqdm(df.iterrows(), desc="Processando Documentos", total=len(df)):
        doc_id = str(index)
        abstract = row["abstract"]
        
        v_doc = g.add_vertex()
        g.vp["name"][v_doc] = doc_id
        g.vp["tipo"][v_doc] = 0
        g.vp["short_term"][v_doc] = ""
        g.vp["posicao"][v_doc] = [-10, doc_y]
        doc_y += 1
        g.vp["size"][v_doc] = 20  # Tamanho menor para termos
        g.vp["amount"][v_doc] = 1
        g.vp["color"][v_doc] = [1.0, 0.0, 0.0, 1.0]  # Vermelho (RGBA)
        doc_vertex[doc_id] = v_doc
        
        doc = nlp(abstract)
        term_freq = {}
        for token in doc:
            if not token.is_stop and not token.is_punct:
                term = token.text.lower().strip()
                if term:
                    term_freq[term] = term_freq.get(term, 0) + 1
        
        for term, freq in term_freq.items():
            term_short = term[:3]
            if term_short not in term_vertex:
                v_term = g.add_vertex()
                g.vp["short_term"][v_term] = term_short
                g.vp["tipo"][v_term] = 1
                g.vp["name"][v_term] = term
                g.vp["color"][v_term] = [0.0, 0.0, 1.0, 1.0]  # Azul (RGBA)
                g.vp["size"][v_term] = 10  # Tamanho menor para termos
                g.vp["posicao"][v_term] = [10,term_y]
                term_y += 1
                g.vp["amount"][v_term] = freq
                term_vertex[term_short] = v_term
            else:
                v_term = term_vertex[term_short]
                g.vp["amount"][v_term] += freq
            
            edge = g.edge(doc_vertex[doc_id], v_term, all_edges=False)
            if edge is None:
                e = g.add_edge(doc_vertex[doc_id], v_term)
                g.ep["weight"][e] = freq
            else:
                g.ep["weight"][edge] += freq
                
    return g


def assign_centered_vertical_positions(vertices, x_fixed, spacing=1.5):
    """
    Posiciona vértices verticalmente, centralizados em torno de y = 0, com x fixo.
    """
    n = len(vertices)
    offset = -(n - 1) / 2 * spacing
    return {v: [x_fixed, offset + i * spacing] for i, v in enumerate(vertices)}



def create_intermediate_graph(g, clusters):
    """
    Cria um grafo intermediário que incorpora clusters de termos entre documentos e termos.
    A ligação direta documento-termo é removida se o termo for absorvido por um cluster.
    """
    # Cria cópia do grafo original com todas as propriedades
    g_inter = g.copy()
    g_inter.vp["cluster_id"] = g_inter.new_vertex_property("int")
    g_inter.vp["cluster"] = g_inter.new_vertex_property("int")

    # Inicializa todos os termos com cluster -1
    for v in g_inter.vertices():
        if int(g_inter.vp["tipo"][v]) == 1:
            g_inter.vp["cluster"][v] = -1

    # Mapeia apenas os termos que foram realmente agrupados
    term_to_cluster = {}
    for cl, term_vertices in clusters.items():
        for v in term_vertices:
            term_to_cluster[g.vp["name"][v]] = cl

    # Atribui os clusters no grafo intermediário com base no nome
    for v in g_inter.vertices():
        if int(g_inter.vp["tipo"][v]) == 1:
            name = g_inter.vp["name"][v]
            g_inter.vp["cluster"][v] = term_to_cluster.get(name, -1)

    # Cria vértices de cluster
    cluster_nodes = {}
    cluster_y = 0
    for cl, term_vertices in clusters.items():
        terms = [(g.vp["name"][v], g.vp["amount"][v]) for v in term_vertices]
        rep_label = " | ".join([t[0] for t in sorted(terms, key=lambda x: x[1], reverse=True)[:3]])

        v_cluster = g_inter.add_vertex()
        g_inter.vp["tipo"][v_cluster] = 2
        g_inter.vp["name"][v_cluster] = rep_label
        g_inter.vp["cluster_id"][v_cluster] = cl
        g_inter.vp["posicao"][v_cluster] = [0, cluster_y]
        g_inter.vp["size"][v_cluster] = 30
        g_inter.vp["color"][v_cluster] = [0.0, 1.0, 0.0, 1.0]
        cluster_nodes[cl] = v_cluster
        cluster_y += 1

    # Conecta clusters aos termos
    for cl, term_vertices in clusters.items():
        cluster_v = cluster_nodes[cl]
        for term_v in term_vertices:
            if g_inter.edge(cluster_v, term_v) is None:
                e = g_inter.add_edge(cluster_v, term_v)
                g_inter.ep["weight"][e] = g.vp["amount"][term_v]

    # Conecta documentos aos clusters, removendo a ligação com termos absorvidos
    for v_doc in g_inter.vertices():
        if int(g_inter.vp["tipo"][v_doc]) != 0:
            continue

        for e in list(v_doc.all_edges()):
            v_term = e.target() if e.source() == v_doc else e.source()
            if int(g_inter.vp["tipo"][v_term]) != 1:
                continue

            cl = int(g_inter.vp["cluster"][v_term])
            if cl == -1:
                continue  # Termo sem cluster: mantém a ligação

            cluster_v = cluster_nodes[cl]

    # spacing_y_docs, spacing_y_clusters, spacing_y_terms = 10.0, 2.5, 3.0

    # docs = [v for v in g_inter.vertices() if int(g_inter.vp["tipo"][v]) == 0]
    # clusters_ = [v for v in g_inter.vertices() if int(g_inter.vp["tipo"][v]) == 2]
    # terms = [v for v in g_inter.vertices() if int(g_inter.vp["tipo"][v]) == 1]

    # positions = {}
    # positions.update(assign_centered_vertical_positions(docs, x_doc, spacing_y_docs))
    # positions.update(assign_centered_vertical_positions(clusters_, x_cluster, spacing_y_clusters))
    # positions.update(assign_centered_vertical_positions(terms, x_term, spacing_y_terms))

    # for v, pos in positions.items():
    #     g_inter.vp["posicao"][v] = pos

    return g_inter

def min_sbm_wew(g):
    """
    Aplica o SBM (Stochastic Block Model) no grafo bipartido DOCUMENTO - TERMOS para identificar comunidades.
    
    :param g: Grafo bipartido com a relação DOCUMENTOS - TERMOS, contendo arestas com pesos.
    :return: Objeto BlockState resultante da aplicação do SBM, representando a partição de comunidades no grafo.
    """
    # Inferindo comunidades usando o SBM de maneira mais simples possível
    state = minimize_blockmodel_dl(g, state_args={"eweight": g.ep["weight"], "pclabel": g.vp["tipo"]})

    # Desenhar as comunidades inferidas com as per'sonalizações
    state.draw(
        pos = sfdp_layout(g),
        vertex_fill_color=g.vp["color"],   # Define a cor dos vértices
        vertex_size=g.vp["size"],          # Define o tamanho dos vértices
        vertex_text=g.vp["name"],         # Define o rótulo dos vértices (ID)
        vertex_text_position = -2,
        vertex_text_color = 'black',
        vertex_font_size=10,  # Tamanho da fonte dos rótulos
        output_size=(800, 800),         # Tamanho da saída
        output="outputs/main/text_graph_sbm.pdf"    # Arquivo PDF de saída
    )

    return state


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
    
    print("\nTermos clusterizados (Word2Vec):")
    for cl, vertices in clusters.items():
        terms = [(g.vp["name"][v], g.vp["amount"][v]) for v in vertices]
        terms_sorted = sorted(terms, key=lambda x: x[1], reverse=True)[:3]
        rep = " | ".join([t[0] for t in terms_sorted])
        print(f"Cluster {cl}: {rep}")
    
    return clusters


def semantic_cohesion(g, clusters, w2v_model):
    """
    Calcula e exibe a coesão semântica de cada cluster com base na similaridade dos vetores Word2Vec dos termos.
    
    :param g: Grafo bipartido com a relação DOCUMENTOS - TERMOS.
    :param clusters: Dicionário de clusters de termos (chave: rótulo do cluster, valor: lista de vértices de termos).
    :param w2v_model: Modelo Word2Vec treinado, usado para acessar os vetores semânticos dos termos.
    :return: Dicionário onde as chaves são os rótulos dos clusters e os valores são as médias de similaridade (coesão) dos termos no cluster.
    """
    print("\nCoesão Semântica dos Clusters:")
    cohesion_scores = {}  # Dicionário para armazenar a coesão de cada cluster

    # Itera sobre cada cluster e seus vértices (termos) presentes no dicionário clusters
    for cl, term_vertices in clusters.items():
        # Cria uma lista com os termos completos de cada vértice do cluster
        terms = [g.vp["name"][v] for v in term_vertices]
        vectors = []  # Lista para armazenar os vetores Word2Vec dos termos

        # Para cada termo, tenta recuperar seu vetor do modelo Word2Vec
        for term in terms:
            try:
                vec = w2v_model.wv[term]
                vectors.append(vec)
            except KeyError:
                # Se o termo não existir no vocabulário do Word2Vec, ignora-o
                continue
        
        # Se houver pelo menos 2 vetores, pode-se calcular a similaridade entre eles
        if len(vectors) >= 2:
            # Calcula a matriz de similaridade entre os vetores usando similaridade do cosseno
            sim_matrix = cosine_similarity(vectors)
            # A coesão é definida como a média de todas as similaridades calculadas
            avg_sim = np.mean(sim_matrix)
            cohesion_scores[cl] = avg_sim  # Armazena a coesão do cluster
            print(f"Cluster {cl}: {avg_sim:.3f}")
        else:
            # Se houver menos de 2 termos com vetor, define a coesão como 0
            cohesion_scores[cl] = 0
    
    return cohesion_scores


def build_block_graph(block_graph, state, g):
    """
    Configura e visualiza o grafo de blocos que representa a conexão entre comunidades de DOCUMENTOS e TERMOS, 
    a partir do resultado da aplicação do SBM.
    
    :param block_graph: Grafo de blocos obtido através do método .bg() no objeto state do SBM.
    :param state: Objeto BlockState resultante da aplicação do SBM no grafo original.
    :param g: Grafo bipartido original com a relação DOCUMENTOS - TERMOS.
    :return: (Não há retorno explícito; a função gera visualizações e altera as propriedades do block_graph.)
    """
    # block_graph = block_graph.copy() # Cópia para rodar o python iterativo
    # Visualizo o grafo de blocos antes das tratativas
    graph_draw(
        block_graph,
        pos=sfdp_layout(block_graph),
        output="outputs/main/text_block_graph_original.pdf"
    )
    
    # Definir propriedades block graph
    name_prop = block_graph.new_vertex_property("string")
    type_prop = block_graph.new_vertex_property("int")
    edge_weight = block_graph.new_edge_property("double")
    number_vertex = block_graph.new_vertex_property("int")
    color_prop = block_graph.new_vertex_property("vector<double>")
    size_prop = block_graph.new_vertex_property("double")
    label_prop = block_graph.new_vertex_property("string")
    vertex_shape = block_graph.new_vertex_property("string")
    pos = block_graph.new_vertex_property("vector<double>")
    block_id_prop = block_graph.new_vertex_property("int")

    
    block_graph.vp["block_id"] = block_id_prop
    block_graph.vp["shape"] = vertex_shape
    block_graph.vp["color"] = color_prop
    block_graph.vp["size"] = size_prop
    block_graph.vp["label"] = label_prop
    block_graph.vp["name"] = name_prop
    block_graph.vp["tipo"] = type_prop
    block_graph.ep["weight"] = edge_weight
    block_graph.vp["nvertex"] = number_vertex
    print(block_graph)  # Print só para confirmar a adição das propriedades

    # Obter a atribuição de blocos original antes da limpeza
    blocks = state.get_blocks().a
    block_sizes = np.bincount(blocks)  

    # Criar um dicionário que associa cada bloco do grafo de blocos aos seus vértices no grafo original
    block_to_vertices = {}
    
    for i in range(len(block_sizes)):  # Iterar sobre os índices dos blocos
        if(block_sizes[i] != 0):
            block_vertices = [v for v in range(len(state.get_blocks().a)) if state.get_blocks().a[v] == i]  
            block_to_vertices[i] = block_vertices  # Salvar no dicionário

            terms = [
                (g.vp["name"][g.vertex(v)], g.vp['amount'][g.vertex(v)]) 
                for v in block_vertices if g.vp["tipo"][g.vertex(v)] == 1
            ]
            docs = sum(1 for v in block_vertices if g.vp["tipo"][g.vertex(v)] == 0)
            
            block_graph.vp["nvertex"][i] = int(block_sizes[i])
            # Aqui você atribui o block id à propriedade recém-criada:
            block_graph.vp["block_id"][i] = i
            
            if terms and docs:
                block_graph.vp["tipo"][i] = 11  # Ambos
            elif terms:
                block_graph.vp["tipo"][i] = 1  # Termos
            elif docs:
                block_graph.vp["tipo"][i] = 0  # Documentos
            else:
                block_graph.vp["tipo"][i] = 22  # Desconhecido

            # Atualiza o nome do bloco para mostrar alguns termos (caso seja um bloco de termos)
            if terms:
                sorted_terms = sorted(terms, key=lambda x: x[1], reverse=True)
                block_graph.vp["name"][i] = "|".join([name for name, _ in sorted_terms[:3]])  # Pega os 3 primeiros nomes

    # Calcular a força das conexões entre blocos
    block_connections = {}  # Dicionário para armazenar a força das conexões entre blocos
    for edge in g.edges():
        source_block = blocks[int(edge.source())]
        target_block = blocks[int(edge.target())]
        if source_block != target_block:  # Ignora arestas dentro do mesmo bloco
            key = tuple(sorted((source_block, target_block)))  # Chave única para o par de blocos
            if key in block_connections:
                block_connections[key] += g.ep["weight"][edge]  # Soma o peso da aresta
            else:
                block_connections[key] = g.ep["weight"][edge]  # Inicializa o peso da aresta

    # Definir a largura das arestas com base na força das conexões
    for edge in block_graph.edges():
        source_block = int(edge.source())
        target_block = int(edge.target())
        key = tuple(sorted((source_block, target_block)))

        if key in block_connections:
            block_graph.ep["weight"][edge] = block_connections[key]  # Define o peso

        # Mapeamento para a largura das arestas (normalização para melhor visualização)
        # block_graph.ep["weight"][edge] = max(1.0, block_graph.ep["weight"][edge] / 75) # Ajuste conforme necessário
        # print(block_graph.ep["weight"][edge])

    # Remove vértices vazios do grafo de blocos
    to_remove = [v for v in block_graph.vertices() if v.out_degree() == 0 and v.in_degree() == 0]
    for v in reversed(to_remove):  # Remover de trás para frente evita problemas de indexação
        block_graph.remove_vertex(v, fast=False)
    # visualize_graph(block_graph, "outputs/main/text_block_graph.pdf")

    # Cria um layout manual
    t = d = 0
    vertices = list(block_graph.vertices())
    for v in tqdm(vertices, desc="Building Block Graph SBM", total=len(vertices)):
        if block_graph.vp["tipo"][v] == 0:  # Documento
            pos[v] = [-2, d]
            d += 1
            block_graph.vp["color"][v] = [1.0, 0.0, 0.0, 1.0]  # Vermelho (RGBA)
            block_graph.vp["size"][v] =  block_graph.vp["nvertex"][v]  # max(20, v.out_degree() * 10)
            
        else:  # Termo
            pos[v] = [2, t]
            t += 1
            block_graph.vp["color"][v] = [0.0, 0.0, 1.0, 1.0]  # Azul (RGBA)
            block_graph.vp["size"][v] = 10  # Tamanho menor para termos

    # Aplicação do SBM ao grafo de blocos
    state_bg = minimize_blockmodel_dl(block_graph)
    # pos = sfdp_layout(block_graph)

    # Agora usamos a propriedade edge_pen_width na visualização
    state_bg.draw(
        pos=pos,
        edge_pen_width= prop_to_size(block_graph.ep["weight"], mi=1, ma=35, power =0.5) ,  # Usa os pesos calculados para definir a largura das arestas block_graph.ep["weight"]
        vertex_fill_color=block_graph.vp["color"],  # Define a cor dos vértices
        vertex_size=prop_to_size(block_graph.vp["size"], mi=20, ma=100),  # Define o tamanho dos vértices block_graph.vp["size"]
        vertex_text=block_graph.vp["name"],  # Exibe os rótulos dos vértices
        vertex_text_position = -2,
        vertex_text_color = 'black',
        vertex_font_size=10,  # Tamanho da fonte dos rótulos
        output_size=(800, 800),  # Tamanho da saída
        output="outputs/main/text_block_graph_sbm.pdf"  # Arquivo PDF de saída
    )

    # return block_to_vertices


def build_document_cluster_edges(g, cluster_nodes):
    """
    Adiciona arestas que conectam documentos a clusters, acumulando os pesos a partir das conexões DOCUMENTO - TERMO.
    
    :param g: Grafo contendo vértices de documentos e termos, com a propriedade 'cluster' definida para termos.
    :param cluster_nodes: Dicionário mapeando o rótulo do cluster para o vértice correspondente no grafo.
    :return: (Não há retorno; as arestas com a propriedade "cluster_weight" são adicionadas ao grafo g.)
    """
    cluster_weight_prop = g.new_edge_property("int")
    g.ep["cluster_weight"] = cluster_weight_prop
    
    for v_doc in g.vertices():
        if int(g.vp["tipo"][v_doc]) != 0:
            continue
        cluster_weights = {}
        for e in v_doc.all_edges():
            v_neighbor = e.target() if e.source() == v_doc else e.source()
            if int(g.vp["tipo"][v_neighbor]) == 1:
                cl = int(g.vp["cluster"][v_neighbor])
                cluster_weights[cl] = cluster_weights.get(cl, 0) + g.ep["weight"][e]
        for cl, weight_sum in cluster_weights.items():
            v_cluster = cluster_nodes.get(cl)
            if v_cluster is not None:
                existing_edge = g.edge(v_doc, v_cluster, all_edges=False)
                if existing_edge is None:
                    new_edge = g.add_edge(v_doc, v_cluster)
                    g.ep["cluster_weight"][new_edge] = weight_sum
                else:
                    g.ep["cluster_weight"][existing_edge] += weight_sum


def count_term_blocks(g, state_wew):
    """
    Conta quantos blocos (comunidades) identificados pelo SBM contêm pelo menos um vértice do tipo TERMO.
    
    :param g: Grafo bipartido com a relação DOCUMENTOS - TERMOS.
    :param state_wew: Objeto BlockState resultante da aplicação do SBM no grafo g.
    :return: Número de blocos que contêm vértices do tipo TERMO.
    """
    blocks = state_wew.get_blocks()  # Mapeia cada vértice ao seu bloco
    blocos_com_termo = set()

    for v in g.vertices():
        if int(g.vp["tipo"][v]) == 1:  # Verifica se o vértice é do tipo termo
            blocos_com_termo.add(blocks[v])  # Adiciona o rótulo do bloco ao conjunto

    return len(blocos_com_termo)


def visualize_docs_and_clusters(g, block_graph, state):
    """
    Cria e retorna um novo grafo que representa a conexão entre comunidades de DOCUMENTOS e clusters de TERMOS.
    
    :param g: Grafo intermediário com a relação DOCUMENTOS - CLUSTERS - TERMOS.
    :param block_graph: Block graph obtido a partir do SBM, contendo propriedades sobre os blocos de documentos.
    :param state: Objeto BlockState resultante da aplicação do SBM no grafo original.
    :return: Novo objeto Graph representando a conexão entre comunidades de DOCUMENTOS e clusters de TERMOS.
    """   
    newG = Graph(directed=False)
    
    # Declaração explícita das propriedades
    name_prop = newG.new_vertex_property("string")
    type_prop = newG.new_vertex_property("int")
    clusterid_prop = newG.new_vertex_property("int")
    blockid_prop = newG.new_vertex_property("int")
    color_prop = newG.new_vertex_property("vector<double>")
    size_prop = newG.new_vertex_property("double")
    edge_weight = newG.new_edge_property("int")
    pos_prop = newG.new_vertex_property("vector<double>")
    
    # Vinculação das propriedades
    newG.vp["name"] = name_prop
    newG.vp["tipo"] = type_prop
    newG.vp["cluster_id"] = clusterid_prop
    newG.vp["block_id"] = blockid_prop
    newG.vp["color"] = color_prop
    newG.vp["size"] = size_prop
    newG.vp["posicao"] = pos_prop
    newG.ep["weight"] = edge_weight

    # --- 1. Adiciona blocos de documentos ---
    # Usamos a propriedade "block_id" do block_graph, que deve ter sido definida anteriormente
    block_vertices = {}
    for v in block_graph.vertices():
        if int(block_graph.vp["tipo"][v]) == 0:
            # Recupera o block id original a partir da propriedade "block_id"
            block_id = int(block_graph.vp["block_id"][v])
            new_v = newG.add_vertex()
            newG.vp["block_id"][new_v] = block_id
            block_vertices[block_id] = new_v
            newG.vp["name"][new_v] = block_graph.vp["name"][v]
            newG.vp["tipo"][new_v] = 0
            newG.vp["color"][new_v] = [1.0, 0.0, 0.0, 1.0]  # Vermelho
            newG.vp["size"][new_v] = block_graph.vp["size"][v]
    
    # --- 2. Adiciona clusters ---
    cluster_vertices = {}
    for v in g.vertices():
        if int(g.vp["tipo"][v]) == 2:
            # Recupera o cluster_id da propriedade "cluster_id" definida no grafo g
            cluster_id = int(g.vp["cluster_id"][v])
            new_v = newG.add_vertex()
            cluster_vertices[cluster_id] = new_v
            newG.vp["name"][new_v] = g.vp["name"][v]
            newG.vp["tipo"][new_v] = 2
            newG.vp["color"][new_v] = [0.0, 1.0, 0.0, 1.0]  # Verde
            newG.vp["size"][new_v] = g.vp["size"][v]
            newG.vp["cluster_id"][new_v] = cluster_id
    
    # --- 3. Conecta blocos a clusters ---
    blocks = state.get_blocks().a  # Vetor onde cada vértice de g tem seu block id
    for doc in g.vertices():
        if int(g.vp["tipo"][doc]) == 0:  # Documentos
            # Recupera o block id do documento; usamos o índice do vértice no vetor blocks
            block_id = int(blocks[int(doc)])
            if block_id in block_vertices:
                for e in doc.all_edges():
                    neighbor = e.target() if e.source() == doc else e.source()
                    if int(g.vp["tipo"][neighbor]) == 2:  # Verifica se o vizinho é um cluster
                        cluster_id = int(g.vp["cluster_id"][neighbor])
                        if cluster_id in cluster_vertices:
                            src = block_vertices[block_id]
                            tgt = cluster_vertices[cluster_id]
                            existing_edge = newG.edge(src, tgt)
                            if existing_edge is None:
                                new_edge = newG.add_edge(src, tgt)
                                newG.ep["weight"][new_edge] = g.ep["weight"][e]
                            else:
                                newG.ep["weight"][existing_edge] += g.ep["weight"][e]
    
    # --- 4. Layout manual ---
    doc_counter = 0
    cluster_counter = 0
    for v in newG.vertices():
        if newG.vp["tipo"][v] == 0:  # Documentos
            newG.vp["posicao"][v] = [-2, doc_counter]
            doc_counter += 1
        else:  # Clusters
            newG.vp["posicao"][v] = [2, cluster_counter]
            cluster_counter += 1

    # # --- 5. Visualização ---
    graph_draw(
        newG,
        pos=newG.vp["posicao"],
        vertex_fill_color=newG.vp["color"],
        vertex_size=prop_to_size(newG.vp["size"], mi=20, ma=100),
        vertex_text=newG.vp["name"],
        vertex_text_position=-2,
        vertex_text_color="black",
        vertex_font_size=10,
        edge_pen_width=prop_to_size(newG.ep["weight"], mi=1, ma=35, power=0.5),
        output_size=(800, 800),
        output="outputs/main/graph_docs_clusters.pdf"
    )

    return newG

def cluster_analyse(clusters, cohesion_scores, g):
    """
    Gera um gráfico e exibe um resumo dos clusters de termos, incluindo um rótulo representativo, quantidade de termos, 
    frequência acumulada e coesão semântica média (o quanto semanticamente próximos são os termos desse clusters).
    
    :param clusters: Dicionário contendo os clusters de termos (chave: ID do cluster, valor: lista de vértices de termos).
    :param cohesion_scores: Dicionário com a coesão semântica média para cada cluster.
    :param g: Grafo intermediário contendo a relação DOCUMENTOS - CLUSTERS - TERMOS.
    :return: (Não há retorno; a função exibe um DataFrame com o resumo dos clusters e gera um gráfico de coesão.)
    """
    cluster_summary = []
    for cl, vertices in clusters.items():
        # Rótulo representativo: 3 termos com maior frequência
        termos = [(g.vp["name"][v], g.vp["amount"][v]) for v in vertices]
        termos_ordenados = sorted(termos, key=lambda x: x[1], reverse=True)
        rep_label = " | ".join([t[0] for t in termos_ordenados[:3]])
        
        # Número de termos e frequência acumulada
        num_termos = len(vertices)
        freq_total = sum(g.vp["amount"][v] for v in vertices)
        
        # Coesão semântica do cluster
        coesao = cohesion_scores.get(cl, 0)
        
        cluster_summary.append({
            "Cluster": cl,
            "Label": rep_label,
            "Num_Termos": num_termos,
            "Freq_Total": freq_total,
            "Coesao": coesao
        })

    df_clusters = pd.DataFrame(cluster_summary)
    print(df_clusters)

    # Ordenando os clusters para visualização
    df_clusters = df_clusters.sort_values("Coesao", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(df_clusters["Label"], df_clusters["Coesao"], color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Rótulo do Cluster (3 termos mais frequentes)")
    plt.ylabel("Coesão Semântica Média")
    plt.title("Coesão Semântica por Cluster (Word2Vec)")
    plt.tight_layout()
    plt.savefig("outputs/main/coesao_clusters.png")  # Salva o gráfico como PNG
    # plt.show()


def calculate_centroid(cluster, w2v_model, g):
    """
    Calcula o centróide de um cluster de termos usando a média dos vetores Word2Vec correspondentes.
    
    :param cluster: Lista de vértices pertencentes ao cluster.
    :param w2v_model: Modelo Word2Vec treinado utilizado para obter os vetores dos termos.
    :param g: Grafo que contém os termos associados aos vértices.
    :return: Vetor (numpy array) representando o centróide do cluster ou None se nenhum vetor foi encontrado.
    """
    vectors = []
    for v in cluster:
        term = g.vp["name"][v]
        try:
            vec = w2v_model.wv[term]
            vectors.append(vec)
        except KeyError:
            continue
    if vectors:
        centroid = np.mean(vectors, axis=0)
        return centroid
    return None


def find_central_term(cluster, centroid, w2v_model, g):
    """
    Retorna os 3 termos mais centrais de um cluster baseado na similaridade com o centróide.
    """
    similarities = []
    centroid = centroid.reshape(1, -1)

    for v in cluster:
        term = g.vp["name"][v]
        try:
            vec = w2v_model.wv[term].reshape(1, -1)
            sim = cosine_similarity(vec, centroid)[0][0]
            similarities.append((term, sim))
        except KeyError:
            continue

    # Ordena todos os termos por similaridade e pega os 3 primeiros
    sorted_terms = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
    terms = [t[0] for t in sorted_terms]
    sims = [t[1] for t in sorted_terms]

    return terms, sims


def find_sbm_block(term, g, state_wew):
    """
    Busca o vértice correspondente a um termo no grafo e retorna o bloco (comunidade) associado segundo o resultado do SBM.
    
    :param term: String representando o termo de interesse.
    :param g: Grafo original onde cada vértice possui a propriedade 'name'.
    :param state_wew: Objeto BlockState resultante do SBM (minimize_blockmodel_dl) aplicado ao grafo.
    :return: Número do bloco (comunidade) associado ao termo ou None se o termo não for encontrado.
    """
    blocks = state_wew.get_blocks().a  # vetor de blocos para cada vértice
    for v in g.vertices():
        if g.vp["name"][v] == term:
            return int(blocks[int(v)])
    return None


def compare_clusters_sbm(clusters, cohesion_scores, g, w2v_model, state_wew):
    """
    Compara os clusters obtidos via Word2Vec com os blocos gerados pelo SBM, gerando um resumo que inclui o termo central e o bloco correspondente.
    
    :param clusters: Dicionário com os clusters de termos (chave: ID do cluster, valor: lista de vértices).
    :param cohesion_scores: Dicionário contendo os valores de coesão semântica para cada cluster.
    :param g: Grafo que contém os termos e suas propriedades.
    :param w2v_model: Modelo Word2Vec treinado para obtenção dos vetores dos termos.
    :param state_wew: Objeto BlockState resultante do SBM aplicado ao grafo original.
    :return: DataFrame contendo um resumo comparativo dos clusters (rótulo, número de termos, frequência, coesão, termo central, similaridade central e bloco SBM).
    """
    summary = []
    for cl, vertices in clusters.items():
        # Rótulo representativo (3 mais frequentes)
        termos = [(g.vp["name"][v], g.vp["amount"][v]) for v in vertices]
        termos_ordenados = sorted(termos, key=lambda x: x[1], reverse=True)
        rep_label = " | ".join([t[0] for t in termos_ordenados[:3]])

        # Frequência e coesão
        num_termos = len(vertices)
        freq_total = sum(g.vp["amount"][v] for v in vertices)
        coesao = cohesion_scores.get(cl, 0)

        # Cálculo do centróide e termos centrais
        centroid = calculate_centroid(vertices, w2v_model, g)
        if centroid is not None:
            central_terms, central_sims = find_central_term(vertices, centroid, w2v_model, g)
        else:
            central_terms, central_sims = [], []

        # Juntar termos centrais e similaridades para exibir como string
        termos_centrais_fmt = " | ".join(central_terms) if central_terms else "None"
        sim_central_fmt = " | ".join(f"{sim:.3f}" for sim in central_sims) if central_sims else "0"

        # Buscar bloco SBM do termo mais central
        if central_terms:
            sbm_block = find_sbm_block(central_terms[0], g, state_wew)
        else:
            sbm_block = None

        summary.append({
            "Cluster ID": cl,
            "Label": rep_label,
            "Num_Termos": num_termos,
            "Freq_Total": freq_total,
            "Coesao": coesao,
            "Termos Centrais": termos_centrais_fmt,
            "Similaridades": sim_central_fmt,
            "SBM Block": sbm_block
        })

    df_summary = pd.DataFrame(summary)
    
    return df_summary


def plot_central_similarity(df_summary):
    """
    Plota um gráfico de barras mostrando a similaridade do termo central com o centróide para cada cluster.
    
    :param df_summary: DataFrame que contém, entre outras colunas, 'Label' e 'Simil Central', resultante da função compare_clusters_sbm.
    :return: (Não há retorno; a função exibe e salva o gráfico gerado.)
    """
    plt.figure(figsize=(10, 6))
    plt.bar(df_summary["Label"], df_summary["Similaridades"], color="coral")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Rótulo do Cluster (3 termos mais frequentes)")
    plt.ylabel("Similaridade do Termo Central com o Centróide")
    plt.title("Similaridade Central dos Clusters")
    plt.tight_layout()
    plt.savefig("outputs/main/similarity_central.png")
    # plt.show()

def compare_partitions_sbm_word2vec(g, state_wew):
    """
    Compara os blocos do SBM com os clusters de termos via métricas da graph_tool.
    Considera apenas os vértices do tipo termo.
    """
    sbm_blocks = []
    w2v_clusters = []

    blocks_array = state_wew.get_blocks().a
    cluster_map = g.vp["cluster"].a  # já é uma numpy array se definido com new_vertex_property

    for v in g.vertices():
        if int(g.vp["tipo"][v]) == 1:  # Apenas TERMOS
            sbm_blocks.append(int(blocks_array[int(v)]))
            w2v_clusters.append(int(cluster_map[int(v)]))

    # Conversão para arrays do Graph-Tool (se necessário)
    sbm_blocks = np.array(sbm_blocks)
    w2v_clusters = np.array(w2v_clusters)

    # Cálculo das métricas

    # Mede o quanto as partições são diferentes. Quanto maior o valor, mais distintas são as segmentações de SBM e Word2Vec.
    vi = variation_information(sbm_blocks, w2v_clusters)  
    # Mede quanta informação uma partição revela sobre a outra. Valores mais altos indicam maior sobreposição entre os agrupamentos.
    mi = mutual_information(sbm_blocks, w2v_clusters)        
    # Retorna três valores: (1) matriz de sobreposição real entre clusters e blocos, (2) sobreposição esperada aleatória e 
    # (3) NMI (Normalized Mutual Information) — valor entre 0 e 1 que resume a similaridade entre as partições.
    po = partition_overlap(sbm_blocks, w2v_clusters)         


    print("\n--- Comparação entre SBM e Word2Vec Clusters ---")
    print(f"Variation of Information: {vi:.4f}")
    print(f"Mutual Information: {mi:.4f}")
    print("Partition Overlap:")
    print(po)  # Isso é uma matriz (overlap + expected overlap + NMI)

    return vi, mi, po

def compute_cluster_purity(clusters, state_wew, g):
    """
    Calcula a pureza de cada cluster em relação aos blocos SBM.
    Pureza = proporção dos termos do cluster que pertencem ao bloco SBM mais comum.
    """
    purities = {}
    blocks = state_wew.get_blocks().a
    summary = []

    for cl, vertices in clusters.items():
        blocos = [blocks[int(v)] for v in vertices]
        count = Counter(blocos)
        total = len(blocos)
        bloco_dominante, freq = count.most_common(1)[0]
        purity = freq / total if total > 0 else 0
        purities[cl] = purity

        summary.append({
            "Cluster": cl,
            "Bloco SBM Dominante": bloco_dominante,
            "Termos no Cluster": total,
            "No mesmo Bloco": freq,
            "Pureza": purity
        })

    df_pureza = pd.DataFrame(summary).sort_values("Pureza", ascending=False)
    print("\n--- Pureza dos Clusters ---")
    print(df_pureza.to_string(index=False))
    return df_pureza


def plot_cluster_sbm_heatmap(clusters, state_wew, g):
    """
    Cria um heatmap da distribuição de termos por cluster em blocos SBM.
    """
    blocks = state_wew.get_blocks().a
    matrix = {}

    for cl, vertices in clusters.items():
        for v in vertices:
            bloco = int(blocks[int(v)])
            if cl not in matrix:
                matrix[cl] = {}
            matrix[cl][bloco] = matrix[cl].get(bloco, 0) + 1

    df_matrix = pd.DataFrame(matrix).fillna(0).astype(int).T  # Transposta: linhas = clusters, colunas = blocos SBM
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_matrix, cmap="Blues", linewidths=0.5, annot=True, fmt="d")
    plt.title("Distribuição de Termos: Clusters × Blocos SBM")
    plt.xlabel("Bloco SBM")
    plt.ylabel("Cluster Word2Vec")
    plt.tight_layout()
    plt.savefig("outputs/main/heatmap_cluster_sbm.png")
    plt.show()

def main():

    start_time = time.time()

    nlp = spacy.load("en_core_web_sm")
    df = pd.read_parquet("wos_sts_journals.parquet")
    df = df.sample(n=300, random_state=42)

    # Definir grid de hiperparâmetros
    param_grid = {
        "vector_size": [50, 100],
        "window": [3, 5, 10],
        "min_count": [1, 2],
        "sg": [0, 1]
    }
    param_list = [
        {"vector_size": vs, "window": w, "min_count": mc, "sg": sg}
        for vs, w, mc, sg in product(
            param_grid["vector_size"],
            param_grid["window"],
            param_grid["min_count"],
            param_grid["sg"]
        )
    ]

    # Podemos escolher duas aproximações:
    # Testar os hiperparametros e definir qual o melhor e inciar o código, essa opção leva mais tempo
    # Utilisar hiperparametros já pré-definidos e fazer o word2vec, dessa forma é mais rápido, porém demanda o conhecimento prévio de quais hiperparametros utilizar
    escolha = input("Deseja realizar o teste de hiperparâmetros? (s/n): ").strip().lower()

    if escolha == 's':
        w2v_model, window = teste_hiperparametros(df, nlp, param_list, param_grid)
    else:
        w2v_model, window = train_word2vec(df, nlp)

    '''
    # Pegar as 5 primeiras palavras do vocabulário
    words = list(w2v_model.wv.index_to_key)[:5]

    # Para cada palavra, imprimir as 5 mais similares
    for word in words:
        print(f"Palavra: {word}")
        print("5 palavras mais similares:")

        # Encontrar as 5 palavras mais similares
        similar_words = w2v_model.wv.most_similar(word, topn=5)
        
        for similar_word, similarity in similar_words:
            print(f"    - {similar_word} (similaridade: {similarity:.4f})")

        print("-" * 50)
    '''
    
    # Construir grafo
    g = initialize_graph()
    g = build_bipartite_graph(g, df, nlp)
    print(g)

    # Salva o grafo original DOCUMENTO - TERMOS
    graph_draw(
    g,
    pos=sfdp_layout(g),           # Layout para posicionar os nós
    # pos = g.vp["posicao"],
    vertex_text=g.vp["name"],     # Usa o rótulo armazenado na propriedade "name"
    vertex_text_position = -2,
    vertex_text_color = 'black',
    vertex_font_size=10,  # Tamanho da fonte dos rótulos
    vertex_fill_color=g.vp["color"],  # Define a cor dos vértices
    output="outputs/main/bipartite_graph.pdf"  # Salva a visualização em PDF
    )

    # Aplicação do sbm com a propriedade de peso nas arestas
    state_wew = min_sbm_wew(g)

    #Construção do grafo de blocos
    block_graph = state_wew.get_bg()
    build_block_graph(block_graph, state_wew, g)

    #Verifica dos blocos gerados, quais são de termos ou não
    num_blocos_termo = count_term_blocks(g, state_wew)
    print(f"De {state_wew.get_nonempty_B()} blocos ao total, {num_blocos_termo} são blocos de termos.")

    # Clusterização com Word2Vec
    clusters = cluster_terms(g, w2v_model, n_clusters=num_blocos_termo)

    # Cohesion score
    cohesion_scores = semantic_cohesion(g, clusters, w2v_model)

    # Comparação entre o número de blocos de termos identificados pelo SBM e os clusters formados via Word2Vec.
    cluster_analyse(clusters, cohesion_scores, g)

    df_comparison = compare_clusters_sbm(clusters, cohesion_scores, g, w2v_model, state_wew)
    df_comparison.to_csv("outputs/main/cluster_sbm_comparison.csv", index=False)
    plot_central_similarity(df_comparison)
    
    # Adicionando os clusters gerados ao grafo original, criando uma entidade intermediária com a relação DOCUMENTO - CLUSTER - TERMOS
    g_intermediate = create_intermediate_graph(g, clusters)

    # Salva o grafo intermediário DOCUMENTO - CLUSTER - TERMOS
    graph_draw(
        g_intermediate,
        # pos = g_intermediate.vp["posicao"],
        pos = sfdp_layout(g_intermediate),
        vertex_text=g_intermediate.vp["name"],
        vertex_text_position = -2,
        vertex_text_color = 'black',
        vertex_font_size=10,  # Tamanho da fonte dos rótulos
        vertex_fill_color=g_intermediate.vp["color"],
        output="outputs/main/grafo_intermediario.pdf"
    )

    
    # Gerar o grafo COMUNIDADE DE DOCUMENTOS - CLUSTERS
    g_doc_clust = visualize_docs_and_clusters(g_intermediate, block_graph, state_wew)
    print(g_doc_clust)  # Grafo comunidade de documentos <-> cluster de termos

    compare_partitions_sbm_word2vec(g, state_wew)

    df_pureza = compute_cluster_purity(clusters, state_wew, g)
    df_pureza.to_csv("outputs/main/pureza_clusters.csv", index=False)
    plot_cluster_sbm_heatmap(clusters, state_wew, g)


    print(f"\nTempo total: {time.time() - start_time:.2f} segundos")

if __name__ == "__main__":
    main()
