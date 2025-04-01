import time
import spacy
import pandas as pd
import numpy as np
from graph_tool.all import (Graph,prop_to_size, graph_draw, sfdp_layout, GraphView, minimize_blockmodel_dl)
from tqdm import tqdm
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

def initialize_graph():

    g = Graph(directed=False)

    name_prop = g.new_vertex_property("string")
    tipo_prop = g.new_vertex_property("int")
    full_term_prop = g.new_vertex_property("string")
    color_prop = g.new_vertex_property("vector<double>")
    posicao_prop = g.new_vertex_property("float")
    amount_prop = g.vp["amount"] = g.new_vertex_property("int")
    size_prop = g.new_vertex_property("double")
    weight_prop = g.new_edge_property("int")

    g.vp["amount"] = amount_prop
    g.vp["size"] = size_prop
    g.vp["color"] = color_prop
    g.vp["name"] = name_prop
    g.vp["tipo"] = tipo_prop
    g.vp["full_term"] = full_term_prop
    g.vp["posicao"] = posicao_prop
    g.ep["weight"] = weight_prop
    
    return g

def train_word2vec(df, nlp):
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
        window=5,       # Número de palavras antes e depois da palavra-alvo consideradas no contexto (janela de contexto)
        min_count=2,    # Ignora palavras que aparecem menos de 2 vezes no corpus
        workers=4       # Número de threads utilizadas para acelerar o treinamento
    )

    return model

def build_bipartite_graph(g, df, nlp):
    doc_vertex = {}
    term_vertex = {}
    
    for index, row in tqdm(df.iterrows(), desc="Processando Documentos", total=len(df)):
        doc_id = str(index)
        abstract = row["abstract"]
        
        v_doc = g.add_vertex()
        g.vp["name"][v_doc] = doc_id
        g.vp["tipo"][v_doc] = 0
        g.vp["full_term"][v_doc] = ""
        g.vp["posicao"][v_doc] = -2
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
                g.vp["name"][v_term] = term_short
                g.vp["tipo"][v_term] = 1
                g.vp["full_term"][v_term] = term
                g.vp["color"][v_term] = [0.0, 0.0, 1.0, 1.0]  # Azul (RGBA)
                g.vp["size"][v_term] = 10  # Tamanho menor para termos
                g.vp["posicao"][v_term] = 2
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

def create_intermediate_graph(g, clusters):

    # Cria uma cópia do grafo original com todas as propriedades
    g_intermediate = g.copy()

    # Mapeia os nós de cluster criados: chave = rótulo do cluster, valor = vértice no grafo intermediário
    cluster_nodes = {}
    
    # Adiciona os nós de cluster com base nos clusters gerados
    for cl, term_vertices in clusters.items():
        # Define o rótulo do cluster com base nos 3 termos mais frequentes
        terms = [(g.vp["full_term"][v], g.vp["amount"][v]) for v in term_vertices]
        terms_sorted = sorted(terms, key=lambda x: x[1], reverse=True)[:3]
        rep_label = " | ".join([t[0] for t in terms_sorted])
        
        # Adiciona o vértice de cluster ao grafo intermediário
        v_cluster = g_intermediate.add_vertex()
        g_intermediate.vp["name"][v_cluster]      = rep_label
        g_intermediate.vp["tipo"][v_cluster]      = 2  # Tipo 2 indica cluster
        g_intermediate.vp["full_term"][v_cluster] = rep_label
        g_intermediate.vp["posicao"][v_cluster]   = 0   # Posicionado no centro
        g_intermediate.vp["size"][v_cluster]      = 30  # Tamanho maior para destaque
        g_intermediate.vp["color"][v_cluster]     = [0.0, 1.0, 0.0, 1.0]  # Verde
        cluster_nodes[cl] = v_cluster

    # Conecta cada nó de cluster aos nós de termos que o compõem.
    for cl, term_vertices in clusters.items():
        cluster_v = cluster_nodes[cl]
        for term_v in term_vertices:
            # Verifica se a aresta já existe para evitar duplicidade
            if g_intermediate.edge(cluster_v, term_v) is None:
                e = g_intermediate.add_edge(cluster_v, term_v)
                # Define o peso da aresta, por exemplo, usando o valor de "amount" do termo
                g_intermediate.ep["weight"][e] = g.vp["amount"][term_v]

    # Conecta documentos (tipo 0) aos clusters correspondentes.
    # Para cada documento, verifica os termos conectados e agrega o peso de ligação com cada cluster.
    for v_doc in g_intermediate.vertices():
        if int(g_intermediate.vp["tipo"][v_doc]) != 0:
            continue  # Considera somente documentos
        cluster_weights = {}
        for e in v_doc.all_edges():
            # Determina o vértice vizinho
            v_neigh = e.target() if e.source() == v_doc else e.source()
            if int(g_intermediate.vp["tipo"][v_neigh]) == 1:
                # Usa a propriedade 'cluster' do vértice de termo (já definida em cluster_terms no grafo original)
                cl_label = int(g_intermediate.vp["cluster"][v_neigh])
                cluster_weights[cl_label] = cluster_weights.get(cl_label, 0) + g_intermediate.ep["weight"][e]
        # Para cada cluster associado ao documento, cria a aresta (se não existir) ou acumula o peso
        for cl_label, weight in cluster_weights.items():
            if cl_label in cluster_nodes:
                cluster_v = cluster_nodes[cl_label]
                if g_intermediate.edge(v_doc, cluster_v) is None:
                    new_e = g_intermediate.add_edge(v_doc, cluster_v)
                    g_intermediate.ep["weight"][new_e] = weight
                else:
                    e = g_intermediate.edge(v_doc, cluster_v)
                    g_intermediate.ep["weight"][e] += weight

    return g_intermediate

def min_sbm_wew(g):

     # #Inferindo comunidades usando o SBM de maneira mais simples possível
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
        output="outputs/text_graph_sbm.pdf"    # Arquivo PDF de saída
    )

    return state

def cluster_terms(g, w2v_model, n_clusters):
    cluster_prop = g.new_vertex_property("int")
    g.vp["cluster"] = cluster_prop

    term_indices = []
    term_vectors = []
    #Percorrer os vértices do grafo e busca no pré-processamento apenas os vetores significantes.
    for v in g.vertices():
        if int(g.vp["tipo"][v]) == 1:
            term = g.vp["full_term"][v]
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
        terms = [(g.vp["full_term"][v], g.vp["amount"][v]) for v in vertices]
        terms_sorted = sorted(terms, key=lambda x: x[1], reverse=True)[:3]
        rep = " | ".join([t[0] for t in terms_sorted])
        print(f"Cluster {cl}: {rep}")
    
    return clusters

def semantic_cohesion(g, clusters, w2v_model):
    print("\nCoesão Semântica dos Clusters:")
    cohesion_scores = {}  # Dicionário para armazenar a coesão de cada cluster

    # Itera sobre cada cluster e seus vértices (termos) presentes no dicionário clusters
    for cl, term_vertices in clusters.items():
        # Cria uma lista com os termos completos de cada vértice do cluster
        terms = [g.vp["full_term"][v] for v in term_vertices]
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
    block_graph = block_graph.copy() # Cópia para rodar o python iterativo
    # Visualizo o grafo de blocos antes das tratativas
    graph_draw(
        block_graph,
        pos=sfdp_layout(block_graph),
        output="outputs/text_block_graph_original.pdf"
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
    # visualize_graph(block_graph, "outputs/text_block_graph.pdf")

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
        edge_pen_width= prop_to_size(block_graph.ep["weight"], mi=1, ma=35, power =1.0) ,  # Usa os pesos calculados para definir a largura das arestas block_graph.ep["weight"]
        vertex_fill_color=block_graph.vp["color"],  # Define a cor dos vértices
        vertex_size=prop_to_size(block_graph.vp["size"], mi=20, ma=100),  # Define o tamanho dos vértices block_graph.vp["size"]
        vertex_text=block_graph.vp["name"],  # Exibe os rótulos dos vértices
        vertex_text_position = -2,
        vertex_text_color = 'black',
        vertex_font_size=10,  # Tamanho da fonte dos rótulos
        output_size=(800, 800),  # Tamanho da saída
        output="outputs/text_block_graph_sbm.pdf"  # Arquivo PDF de saída
    )

    # return block_to_vertices


def add_cluster_nodes(g, clusters):
    g.vp["cluster_id"] = g.new_vertex_property("int")
    cluster_nodes = {}
    for cl, term_vertices in clusters.items():
        terms = [(g.vp["full_term"][v], g.vp["amount"][v]) for v in term_vertices]
        terms_sorted = sorted(terms, key=lambda x: x[1], reverse=True)[:3]
        rep_label = " | ".join([t[0] for t in terms_sorted])
        
        v_cluster = g.add_vertex()
        g.vp["name"][v_cluster] = rep_label
        g.vp["tipo"][v_cluster] = 2
        g.vp["full_term"][v_cluster] = rep_label
        g.vp["posicao"][v_cluster] = 2
        g.vp["cluster_id"][v_cluster] = cl
        cluster_nodes[cl] = v_cluster

    return cluster_nodes

def build_document_cluster_edges(g, cluster_nodes):
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

    blocks = state_wew.get_blocks()  # Mapeia cada vértice ao seu bloco
    blocos_com_termo = set()

    for v in g.vertices():
        if int(g.vp["tipo"][v]) == 1:  # Verifica se o vértice é do tipo termo
            blocos_com_termo.add(blocks[v])  # Adiciona o rótulo do bloco ao conjunto

    return len(blocos_com_termo)


def visualize_docs_and_clusters(g, cohesion_scores, output_file="outputs/docs_clusters.pdf"):
    g_view = GraphView(g, vfilt=lambda v: int(g.vp["tipo"][v]) in [0, 2])
    
    docs = [v for v in g_view.vertices() if int(g.vp["tipo"][v]) == 0]
    clus = [v for v in g_view.vertices() if int(g.vp["tipo"][v]) == 2]
    
    pos = g_view.new_vertex_property("vector<double>")
    
    # Posicionamento dos nós
    n_docs = len(docs)
    for i, v in enumerate(sorted(docs, key=lambda v: g.vp["name"][v])):
        y = - (i - n_docs/2)
        pos[v] = [-2, y]
    
    n_clus = len(clus)
    for i, v in enumerate(sorted(clus, key=lambda v: g.vp["name"][v])):
        y = - (i - n_clus/2)
        pos[v] = [2, y]
    
    # --- CORREÇÃO DEFINITIVA ---
    vertex_text_dist = g_view.new_vertex_property("double")  # Renomeado para evitar conflito
    for v in g_view.vertices():
        if int(g.vp["tipo"][v]) == 2:
            vertex_text_dist[v] = 10  # Clusters
        else:
            vertex_text_dist[v] = 3    # Documentos
    
    # Cores dinâmicas
    vertex_colors = g_view.new_vertex_property("vector<double>")
    for v in g_view.vertices():
        if int(g.vp["tipo"][v]) == 0:
            vertex_colors[v] = [1, 0, 0, 1]  # Vermelho para documentos
        else:
            cl = int(g.vp["cluster_id"][v])
            intensity = cohesion_scores.get(cl, 0)
            vertex_colors[v] = [0, intensity, 0, 1]  # Verde para clusters
    
    # Desenho do grafo
    graph_draw(
        g_view,
        pos=pos,
        vertex_fill_color=vertex_colors,
        vertex_size=20,
        vertex_text=g_view.vp["name"],
        vertex_text_position = -2,
        vertex_text_color = 'black',
        vertex_font_size=10,  # Tamanho da fonte dos rótulos
        output=output_file
    )


def main():
    start_time = time.time()
    
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_parquet("wos_sts_journals.parquet")
    df = df.sample(n=300, random_state=42)
    
    # Treinar Word2Vec
    w2v_model = train_word2vec(df, nlp)

    # # Pegar as 5 primeiras palavras do vocabulário
    # words = list(w2v_model.wv.index_to_key)[:5]

    # # Para cada palavra, imprimir as 5 mais similares
    # for word in words:
    #     print(f"Palavra: {word}")
    #     print("5 palavras mais similares:")

    #     # Encontrar as 5 palavras mais similares
    #     similar_words = w2v_model.wv.most_similar(word, topn=5)
        
    #     for similar_word, similarity in similar_words:
    #         print(f"    - {similar_word} (similaridade: {similarity:.4f})")

    #     print("-" * 50)
    
    # Construir grafo
    g = initialize_graph()
    g = build_bipartite_graph(g, df, nlp)
    print(g)

    graph_draw(
    g,
    pos=sfdp_layout(g),           # Layout para posicionar os nós
    vertex_text=g.vp["name"],     # Usa o rótulo armazenado na propriedade "name"
    vertex_text_position = -2,
    vertex_text_color = 'black',
    vertex_font_size=10,  # Tamanho da fonte dos rótulos
    vertex_fill_color=g.vp["color"],  # Define a cor dos vértices
    output="outputs/bipartite_graph.pdf"  # Salva a visualização em PDF
    )
    # print("\nEtapa 1: Total de nós:", g.num_vertices())
    # print("Etapa 1: Arestas:", g.num_edges())

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
    cohesion_scores = semantic_cohesion(g, clusters, w2v_model)
    
    #Adicionando o cluster gerados ao grafo original, para criar uma entidade intermediária tendo a relação DOCUMENTO - CLUSTER - TERMOS
    
    # #Aonde o label de cada cluster será os 3 primeiros termos mais frequentes
    # cluster_nodes = add_cluster_nodes(g, clusters)  #Função comentada por hora, pois essa operação é feita ao desenhar o grafo intermidiário.
    
    # Cria o grafo intermediário com os nós de cluster centralizados
    g_intermediate = create_intermediate_graph(g, clusters)

    # Salva e/ou visualiza o grafo intermediário
    graph_draw(
        g_intermediate,
        vertex_text=g_intermediate.vp["name"],
        vertex_text_position = -2,
        vertex_text_color = 'black',
        vertex_font_size=10,  # Tamanho da fonte dos rótulos
        vertex_fill_color=g_intermediate.vp["color"],
        output="outputs/grafo_intermediario.pdf"
    )

    print("\nEtapa 3: Total de nós:", g.num_vertices())
    exit()
    
    visualize_docs_and_clusters(g, cohesion_scores)
    print(f"\nTempo total: {time.time() - start_time:.2f} segundos")

if __name__ == "__main__":
    main()