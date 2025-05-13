import time
import spacy
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Usa backend para salvar arquivos, sem abrir janelas
from graph_tool.all import (Graph, prop_to_size, graph_draw, sfdp_layout, minimize_blockmodel_dl, variation_information, mutual_information, partition_overlap)
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def initialize_graph():
    """
    Inicializa e configura um grafo não direcionado com as propriedades básicas baseado no projeto Sashimi.
    
    :return: Um objeto Graph com propriedades de vértice e de aresta definidas.
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

def build_window_graph(g, df, nlp, w):
    """
    Cria o grafo tripartido Documento – Janela – Termo.
    Se w == "full", a janela cobre o abstract completo.
    """
    g.vp["termos"] = g.new_vertex_property("object")   # lista de tokens da janela

    window_vertex = {}          # (doc_id, tuple(win_tokens)) → vértice-janela
    term_vertex   = {}          # termo → vértice
    doc_vertex    = {}
    doc_y = term_y = win_y = 0

    for idx, row in tqdm(df.iterrows(), desc="Processando Doc-Jan-Termo", total=len(df)):
        doc_id   = str(idx)
        abstract = row["abstract"]

        # ─────────── vértice Documento ───────────
        v_doc = g.add_vertex()
        g.vp["name"][v_doc]    = doc_id
        g.vp["tipo"][v_doc]    = 0
        g.vp["posicao"][v_doc] = [-15, doc_y]; doc_y += 1
        g.vp["size"][v_doc]    = 20
        g.vp["color"][v_doc]   = [1, 0, 0, 1]
        doc_vertex[doc_id]     = v_doc

        # tokenização
        doc_spacy = nlp(abstract)
        tokens = [tok.text.lower().strip()
                  for tok in doc_spacy
                  if not tok.is_stop and not tok.is_punct]

        # ─── ajuste: w == "full" ⇒ janela = abstract inteiro ───
        if w == "full":
            w_local = len(tokens)            # maior que qualquer índice
        else:
            w_local = w                      # usa valor inteiro fornecido


        # loop sobre posições (centro da janela)
        for i in range(len(tokens)):
            start = max(0, i - w_local)
            end   = min(len(tokens), i + w_local + 1)
            win_tokens = tokens[start:end]

            win_id = (doc_id, tuple(win_tokens))   # identifica janela pelo conteúdo

            # criar vértice-Janela, se novo
            if win_id not in window_vertex:
                v_win = g.add_vertex()
                window_vertex[win_id]  = v_win
                g.vp["name"][v_win]    = tokens[i]        # termo central (label)
                g.vp["tipo"][v_win]    = 3
                g.vp["posicao"][v_win] = [0, win_y]; win_y += 1
                g.vp["size"][v_win]    = 12
                g.vp["color"][v_win]   = [0, 0.7, 0, 1]
                g.vp["termos"][v_win]  = win_tokens       # guarda lista de termos

            v_win      = window_vertex[win_id]
            freq_map   = Counter(win_tokens)
            total_freq = 0

            # ─── liga Janela → Termo ───
            for term, freq in freq_map.items():
                if term not in term_vertex:
                    v_term = g.add_vertex()
                    term_vertex[term] = v_term
                    g.vp["name"][v_term]       = term
                    g.vp["short_term"][v_term] = term[:3]
                    g.vp["tipo"][v_term]       = 1
                    g.vp["posicao"][v_term]    = [15, term_y]; term_y += 1
                    g.vp["size"][v_term]       = 10
                    g.vp["color"][v_term]      = [0, 0, 1, 1]
                    g.vp["amount"][v_term]     = freq
                else:
                    v_term = term_vertex[term]
                    g.vp["amount"][v_term] += freq

                edge = g.edge(v_win, v_term, all_edges=False)
                if edge is None:
                    e = g.add_edge(v_win, v_term)
                    g.ep["weight"][e] = freq
                else:
                    g.ep["weight"][edge] += freq

                total_freq += freq

            # ─── liga Documento → Janela ───
            if g.edge(v_doc, v_win) is None:
                e_dj = g.add_edge(v_doc, v_win)
                g.ep["weight"][e_dj] = total_freq

    return g


def min_sbm_docs_janelas(g_doc_jan):
    """
    Aplica o SBM (Stochastic Block Model) no gafo bipartido DOCUMENTO - JANELAS para identificar comunidades.
    
    :param g_doc_jan: Grafo bipartido com a relação DOCUMENTOS - JANELAS, contendo arestas com pesos.
    :return: Objeto BlockState resultante da aplicação do SBM, representando a partição de comunidades no grafo.
    """
    # Inferindo comunidades usando o SBM de maneira mais simples possível
    state = minimize_blockmodel_dl(g_doc_jan, state_args={"eweight": g_doc_jan.ep["weight"], "pclabel": g_doc_jan.vp["tipo"]})

    # # Desenhar as comunidades inferidas com as per'sonalizações
    # state.draw(
    #     pos = sfdp_layout(g_doc_jan),
    #     vertex_fill_color=g_doc_jan.vp["color"],   # Define a cor dos vértices
    #     vertex_size=g_doc_jan.vp["size"],          # Define o tamanho dos vértices
    #     vertex_text=g_doc_jan.vp["name"],         # Define o rótulo dos vértices (ID)
    #     vertex_text_position = -2,
    #     vertex_text_color = 'black',
    #     vertex_font_size=10,  # Tamanho da fonte dos rótulos
    #     output_size=(800, 800),         # Tamanho da saída
    #     output="outputs/window/text_graph_sbm.pdf"    # Arquivo PDF de saída
    # )

    return state

def build_block_graph(block_graph, state, g, window):
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
        output="outputs/window/text_block_graph_original.pdf"
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
                block_graph.vp["tipo"][i] = 3  # Janela
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
    # visualize_graph(block_graph, "outputs/window/text_block_graph.pdf")

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
        output="outputs/window/text_block_window_graph_sbm"+ str(window) +".pdf"  # Arquivo PDF de saída
    )

    # return block_to_vertices

def count_jan_blocks(g, state_wew):
    """
    Conta quantos blocos (comunidades) identificados pelo SBM contêm pelo menos um vértice do tipo JANELA.
    
    :param g: Grafo bipartido com a relação DOCUMENTOS - JANELA.
    :param state_wew: Objeto BlockState resultante da aplicação do SBM no grafo g.
    :return: Número de blocos que contêm vértices do tipo JANELA.
    """
    blocks = state_wew.get_blocks()
    block_to_types = {}

    for v in g.vertices():
        bloco = blocks[v]
        tipo = g.vp["tipo"][v]
        if bloco not in block_to_types:
            block_to_types[bloco] = set()
        block_to_types[bloco].add(tipo)

    # Contar blocos que têm apenas termos (tipo 3)
    blocos_de_termos = [b for b, tipos in block_to_types.items() if tipos == {3}]
    return len(blocos_de_termos)

def extract_doc_jan_graph(g):
    """
    Cria um novo grafo contendo apenas os vértices tipo 0 (documentos) e 3 (janelas)
    e as arestas que os conectam.
    """
    g_doc_jan = Graph(directed=False)
    
    # Copiar propriedades necessárias
    for prop in g.vp.keys():
        g_doc_jan.vp[prop] = g_doc_jan.new_vertex_property(g.vp[prop].value_type())
    for prop in g.ep.keys():
        g_doc_jan.ep[prop] = g_doc_jan.new_edge_property(g.ep[prop].value_type())

    # Mapear vértices válidos
    vertex_map = {}
    for v in g.vertices():
        if g.vp["tipo"][v] in (0, 3):
            v_new = g_doc_jan.add_vertex()
            vertex_map[int(v)] = v_new
            for prop in g.vp.keys():
                g_doc_jan.vp[prop][v_new] = g.vp[prop][v]

    # Adicionar apenas arestas entre os vértices válidos
    for e in g.edges():
        src, tgt = int(e.source()), int(e.target())
        if src in vertex_map and tgt in vertex_map:
            e_new = g_doc_jan.add_edge(vertex_map[src], vertex_map[tgt])
            for prop in g.ep.keys():
                g_doc_jan.ep[prop][e_new] = g.ep[prop][e]

    return g_doc_jan


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
    
    # print("\nTermos clusterizados (Word2Vec):")
    # for cl, vertices in clusters.items():
    #     terms = [(g.vp["name"][v], g.vp["amount"][v]) for v in vertices]
    #     terms_sorted = sorted(terms, key=lambda x: x[1], reverse=True)[:3]
    #     rep = " | ".join([t[0] for t in terms_sorted])
    #     print(f"Cluster {cl}: {rep}")
    
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
        
        if len(vectors) >= 2:
            sim_matrix = cosine_similarity(vectors)
            n = sim_matrix.shape[0]
            triu_indices = np.triu_indices(n, k=1)
            sims = sim_matrix[triu_indices]
            avg_sim = np.mean(sims)
            cohesion_scores[cl] = avg_sim
            # print(f"Cluster {cl}: {avg_sim:.3f}")
        else:
            cohesion_scores[cl] = 0

    
    return cohesion_scores

def cluster_analyse(clusters, cohesion_scores, g, window):
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

    # # Ordenando os clusters para visualização
    # df_clusters = df_clusters.sort_values("Coesao", ascending=False)

    # plt.figure(figsize=(10, 6))
    # plt.bar(df_clusters["Label"], df_clusters["Coesao"], color="skyblue")
    # plt.xticks(rotation=45, ha="right")
    # plt.xlabel("Rótulo do Cluster (3 termos mais frequentes)")
    # plt.ylabel("Coesão Semântica Média")
    # plt.title("Coesão Semântica por Cluster (Word2Vec)")
    # plt.tight_layout()
    # plt.savefig("outputs/window/coesao_w"+ str(window) +"_clusters.png")  # Salva o gráfico como PNG
    # # plt.show()

def train_word2vec(df, nlp, window):
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
        window=window,       # Número de palavras antes e depois da palavra-alvo consideradas no contexto (janela de contexto)
        min_count=2,    # Ignora palavras que aparecem menos de 2 vezes no corpus
        sg=0,           # 1 para skip-gram ou 0 (default) para CBOW. CBOW: contexto ➜ palavra | Skip‑gram: palavra ➜ contexto
        workers=4       # Número de threads utilizadas para acelerar o treinamento
    )
 
    return model

def draw_base_graphs(g, g_doc_jan, g_doc_term,window):
    window = str(window)
    # Salva o grafo original DOCUMENTO - JANELAS - TERMOS
    graph_draw(
    g,
    pos=sfdp_layout(g),           # Layout para posicionar os nós
    # pos = g.vp["posicao"],
    vertex_text=g.vp["name"],     # Usa o rótulo armazenado na propriedade "name"
    vertex_text_position = -2,
    vertex_text_color = 'black',
    vertex_font_size=10,  # Tamanho da fonte dos rótulos
    vertex_fill_color=g.vp["color"],  # Define a cor dos vértices
    output="outputs/window/window" + window + "_graph_d-j-t.pdf"  # Salva a visualização em PDF
    )

    # Salva o grafo original DOCUMENTO - JANELAS
    graph_draw(
    g_doc_jan,
    pos=sfdp_layout(g_doc_jan),           # Layout para posicionar os nós
    # pos = g_doc_jan.vp["posicao"],
    vertex_text=g_doc_jan.vp["name"],     # Usa o rótulo armazenado na propriedade "name"
    vertex_text_position = -2,
    vertex_text_color = 'black',
    vertex_font_size=10,  # Tamanho da fonte dos rótulos
    vertex_fill_color=g_doc_jan.vp["color"],  # Define a cor dos vértices
    output="outputs/window/window" + window + "_graph_d-j.pdf"  # Salva a visualização em PDF
    )

    # Salva o grafo original DOCUMENTO - TERMOS
    graph_draw(
    g_doc_term,
    pos=sfdp_layout(g_doc_term),           # Layout para posicionar os nós
    # pos = g_doc_term.vp["posicao"],
    vertex_text=g_doc_term.vp["name"],     # Usa o rótulo armazenado na propriedade "name"
    vertex_text_position = -2,
    vertex_text_color = 'black',
    vertex_font_size=10,  # Tamanho da fonte dos rótulos
    vertex_fill_color=g_doc_term.vp["color"],  # Define a cor dos vértices
    output="outputs/window/window" + window + "_graph_d-t.pdf"  # Salva a visualização em PDF
    )

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
            existing = g_inter.edge(v_doc, cluster_v)
            if existing is None:
                new_e = g_inter.add_edge(v_doc, cluster_v)
                g_inter.ep["weight"][new_e] = g_inter.ep["weight"][e]
            else:
                g_inter.ep["weight"][existing] += g_inter.ep["weight"][e]

            g_inter.remove_edge(e)  # Remove ligação direta doc-termo

    return g_inter

def compare_partitions_sbm_word2vec(g_clusters, state_sbm, g_doc_jan):
    """
    Compara os clusters do Word2Vec (em g_clusters = g_doc_term)
    com os blocos SBM de janelas (em g_doc_jan),
    reconstruindo term→bloco via a propriedade 'termos' das janelas.
    """
    # 1) obtém o vetor de blocos do SBM (aplicado a g_doc_jan)
    blocks_array = state_sbm.get_blocks().a

    # 2) constrói mapa term_name → bloco_SBM
    term_to_block = {}
    for v in g_doc_jan.vertices():
        # só janelas (tipo==3)
        if int(g_doc_jan.vp["tipo"][v]) != 3:
            continue
        bloco = int(blocks_array[int(v)])  # índice seguro: v pertence a g_doc_jan
        # a propriedade 'termos' é a lista de strings
        for term in g_doc_jan.vp["termos"][v]:
            term_to_block[term] = bloco

    # 3) agora itera sobre os termos em g_clusters (g_doc_term)
    sbm_blocks   = []
    w2v_clusters = []

    for v in g_clusters.vertices():
        # só termos (tipo==1)
        if int(g_clusters.vp["tipo"][v]) != 1:
            continue
        term = g_clusters.vp["name"][v]
        # só mantém se apareceu em alguma janela
        if term not in term_to_block:
            continue

        sbm_blocks.append(term_to_block[term])
        w2v_clusters.append(int(g_clusters.vp["cluster"][v]))

    if not sbm_blocks:
        raise ValueError("Nenhum termo comum encontrado entre janelas SBM e clusters Word2Vec.")

    sbm_blocks   = np.array(sbm_blocks)
    w2v_clusters = np.array(w2v_clusters)

    # 4) cálculo das métricas
    vi = variation_information(sbm_blocks, w2v_clusters)
    mi = mutual_information  (sbm_blocks, w2v_clusters)
    po = partition_overlap   (sbm_blocks, w2v_clusters)

    print("\n--- Comparação SBM (via janelas) × Word2Vec Clusters ---")
    print(f"Variation of Information: {vi:.4f}")
    print(f"Mutual Information:    {mi:.4f}")
    print("Partition Overlap:")
    print(po)

    return vi, mi, po

def plot_cohesion_relative_to_window(csv_path="outputs/window/results_window.csv",
                                     save_path="outputs/window/cohesion_relative.png",
                                     show=True):
    # Lê o CSV
    df = pd.read_csv(csv_path)

    # Trata a janela FULL como último item
    def sort_key(w):
        return int(w) if w != "FULL" else 70

    df["sort_key"] = df["window"].apply(sort_key)
    df = df.sort_values("sort_key")

    # Dados
    x_labels = df["window"].astype(str).values
    x_numeric = df["sort_key"].values
    cohesion = df["mean_cohesion"].values
    projection_y = cohesion * x_numeric
    cohesion_percent = cohesion * 100

    # Cria dois subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Subplot 1: Projeção na diagonal ---
    lim = int(1.1 * max(x_numeric))
    ax1.plot([0, lim], [0, lim], 'k--', label="Coesão Máxima (y = x)")

    for i in range(len(x_numeric)):
        ax1.scatter(x_numeric[i], projection_y[i], s=80)
        ax1.text(x_numeric[i] + 3, projection_y[i], x_labels[i], fontsize=9)

    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel("Tamanho da Janela de Contexto")
    ax1.set_ylabel("Coesão × Janela")
    ax1.set_title("Projeção de Coesão sobre a Janela de Contexto")
    ax1.grid(True)

    # --- Subplot 2: Barras de coesão em % ---
    bars = ax2.bar(x_labels, cohesion_percent, color="skyblue", edgecolor="black")
    for bar, val in zip(bars, cohesion_percent):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%", 
                 ha="center", va="bottom", fontsize=9)

    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Coesão Semântica (%)")
    ax2.set_xlabel("Tamanho da Janela de Contexto")
    ax2.set_title("Coesão Média por Janela")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    # Finalização
    plt.tight_layout()
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()



def run_pipeline(df, nlp, win):
    """
    Executa o fluxo completo para um dado tamanho de janela (win):
    grafo DOC-JAN-TERMO  →  SBM em DOC-JAN  →  Word2Vec+clusters em DOC-TERMO
    Retorna um dicionário com métricas resumidas.
    """
    # 1) grafo tripartido
    g = initialize_graph()
    g= build_window_graph(g, df, nlp, win)
    print("Grafo DOC-JAN-TERM")
    print(g)

    # 2) sub-grafos bipartidos
    g_doc_jan  = extract_doc_jan_graph(g)
    print("Grafo DOC-JAN")
    print(g_doc_jan)

    g_doc_term = extract_doc_term_graph(g)
    print("Grafo DOC-TERM")
    print(g_doc_term)

    # # Impressão dos 3 grafos bases do projeto
    # draw_base_graphs(g,g_doc_jan,g_doc_term, win)

    # 3) SBM em DOC-JAN
    state = min_sbm_docs_janelas(g_doc_jan)
    n_blocks_jan = count_jan_blocks(g_doc_jan, state)
        

    # 4) Word2Vec (usa janela int; se 'full', põe 10000, pois o gensim trata isso automaticamente e limita ao valor máximo de termos do abstract
    w_int = win if isinstance(win, int) else 10000
    w2v_model = train_word2vec(df, nlp, w_int)
    window_label = "FULL" if win == "full" else w_int

    # # #Construção do grafo de blocos
    # block_graph = state.get_bg()
    # build_block_graph(block_graph, state, g_doc_jan, win) #TODO preciso adaptar essa função para a milha lógica

    # 5) clusters em DOC-TERMO
    clusters = cluster_terms(g_doc_term, w2v_model, n_clusters=n_blocks_jan)
    cohesion_scores = semantic_cohesion(g_doc_term, clusters, w2v_model)
    mean_cohesion = np.mean(list(cohesion_scores.values())) if cohesion_scores else 0.0

    # 5.1) Comparação entre o número de blocos de termos identificados pelo SBM e os clusters formados via Word2Vec.
    cluster_analyse(clusters, cohesion_scores, g_doc_term, win)

    # # Adicionando os clusters gerados ao grafo original, criando uma entidade intermediária com a relação DOCUMENTO - CLUSTER - TERMOS
    # g_intermediate = create_intermediate_graph(g_doc_term, clusters)

    df_comparison = compare_clusters_sbm(clusters, cohesion_scores, g_doc_jan, w2v_model, state)
    df_comparison.to_csv("outputs/window/cluster_sbm_w"+ str(window_label) +"_comparison.csv", index=False)


    # 6) métricas SBM × Word2Vec
    vi, mi, po = compare_partitions_sbm_word2vec(
        g_doc_term,  # grafo DOC-TERM com clusters
        state,       # resultado do SBM em g_doc_jan
        g_doc_jan    # grafo DOC-JAN onde o SBM foi aplicado
    )

    nmi = po[2] if isinstance(po, (list, tuple, np.ndarray)) else po

    return {
        "window": window_label,
        "blocks": n_blocks_jan,
        "clusters": len(clusters),
        "VI": vi,
        "NMI": nmi,
        "mean_cohesion": mean_cohesion
    }


def main():
    start_time = time.time()

    nlp = spacy.load("en_core_web_sm")
    df  = pd.read_parquet("wos_sts_journals.parquet").sample(n=300, random_state=42)

    WINDOW_LIST = [5, 10, 20, 40, 50, "full"]   # último = abstract inteiro
    results = []

    for win in WINDOW_LIST:
        print("\n" + "="*60)
        print(f" PIPELINE  |  window = {win}")
        print("="*60)
        res = run_pipeline(df, nlp, win)
        results.append(res)

    # salva resumo
    df_results = pd.DataFrame(results)
    df_results.to_csv("outputs/window/results_window.csv", index=False) 
    print("\nResumo final:\n", df_results)

    #Impressão do gráfico que avalia a relação entre SBM x Word2Vec+Kmeans    
    plot_cohesion_relative_to_window()

    print(f"\nTempo total: {time.time() - start_time:.2f} s")

if __name__ == "__main__":
    main()
