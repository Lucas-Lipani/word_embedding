from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
from gensim.models import Word2Vec


def train_word2vec(df, nlp, window):
    """
    Processa os abstracts do corpus e treina um modelo Word2Vec usando os tokens
    obtidos.

    :param df: DataFrame contendo uma coluna "abstract" com os textos a serem
    processados.
    :param nlp: Tokenizer do spaCy (por exemplo, "en_core_web_sm") para processar os
    textos e remover stop words e pontuações.
    :param window: Tamanho da janela de conteto para o treinamento do modelo.
    :return: Um modelo Word2Vec treinado com os tokens extraídos dos abstracts.
    """

    print(
        f"\nIniciando treinamento do modelo Word2Vec com janela de contexto = {window}"
    )

    sentences = df["tokens"].tolist()

    model = Word2Vec(
        sentences,
        vector_size=100,
        window=window,
        min_count=1,
        sg=1,
        workers=4,
        epochs=15,
    )

    print(f"Treinamento concluído para janela de contexto = {window}.")

    return model


def get_or_train_w2v_model(w2v_models, window_size, df_docs, nlp):
    """
    Retorna modelo Word2Vec treinado para a janela fornecida.
    Se já existir no dicionário, reutiliza.
    """
    if window_size not in w2v_models:
        w_int = 10000 if window_size == "full" else int(window_size)
        model = train_word2vec(df_docs, nlp, w_int)
        w2v_models[window_size] = model
    return w2v_models[window_size]


def kmeans_clustering(g, n_clusters, term_vectors):
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


def cluster_terms(g, w2v_model, n_clusters):
    """
    Realiza a clusterização dos termos do grafo utilizando os vetores semânticos do
    modelo Word2Vec.

    :param g: Grafo bipartido com a relação DOCUMENTOS - TERMOS.
    :param w2v_model: Modelo Word2Vec previamente treinado, usado para extrair vetores
    semânticos de termos.
    :param n_clusters: Número de clusters a serem formados, geralmente definido a partir
    do número de comunidades identificadas pelo SBM.
    :return: Dicionário onde as chaves são os rótulos dos clusters e os valores são
    listas de vértices (termos) que pertencem a cada cluster.
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

    clusters = kmeans_clustering(g, n_clusters, term_vectors)

    return clusters
