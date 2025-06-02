from graph_tool.all import (Graph)
from tqdm import tqdm
import graph_build
from collections import Counter

def initialize_graph():
    """
    Inicializa e configura um grafo não direcionado com as propriedades básicas baseado
    no projeto Sashimi.

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
        # abstract = ("Janela de teste para analisar se está fazendo tudo certo, caso "
        # "esteja tudo certo, irei analisar o próximo.")

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


def extract_context_window_term_graph(g_jan_term):
    """
    Constrói grafo tripartido:
      termo como contexto (tipo 4) → janela (tipo 3) → termo central (tipo 1)

    Arestas originais de layer=1 (janela–termo contexto) são removidas.
    Para cada uma, cria-se um novo vértice tipo 4 (contextualização do termo),
    que será ligado à janela via layer=0.

    As posições são definidas por tipo:
      tipo 4 → x=-15, tipo 3 → x=0, tipo 1 → x=15
    """
    g = graph_build.initialize_graph()

    cont_y = win_y = term_y = 0

    # mapas: termo → v_term, janela → v_win, termo_contexto_nome → v_ctx
    term_map = {}
    win_map = {}
    ctx_map = {}
    g.vp["termos"] = g.new_vertex_property("object")

    for v in g_jan_term.vertices():
        tipo = int(g_jan_term.vp["tipo"][v])
        nome = g_jan_term.vp["name"][v]

        if tipo == 1:
            v_term = g.add_vertex()
            g.vp["name"][v_term] = nome
            g.vp["tipo"][v_term] = 1
            g.vp["posicao"][v_term] = [15, term_y]
            g.vp["color"][v_term] = [0, 0, 1, 1]
            g.vp["size"][v_term] = 10
            term_y += 1
            term_map[int(v)] = v_term

        elif tipo == 3:
            v_win = g.add_vertex()
            g.vp["name"][v_win] = nome
            g.vp["tipo"][v_win] = 3
            g.vp["posicao"][v_win] = [0, win_y]
            g.vp["color"][v_win] = [0.6, 0.6, 0.6, 1]
            g.vp["size"][v_win] = 15
            win_y += 1
            win_map[int(v)] = v_win

    # percorre arestas
    for e in g_jan_term.edges():
        layer = int(g_jan_term.ep["layer"][e])
        v1, v2 = e.source(), e.target()
        tipo1 = int(g_jan_term.vp["tipo"][v1])
        tipo2 = int(g_jan_term.vp["tipo"][v2])
        peso = int(g_jan_term.ep["weight"][e])

        # caso layer = 0 → Janela ↔ Termo central → mantém igual
        if layer == 0 and {tipo1, tipo2} == {1, 3}:
            id_win = int(v1) if tipo1 == 3 else int(v2)
            id_term = int(v1) if tipo1 == 1 else int(v2)
            v_win = win_map[id_win]
            v_term = term_map[id_term]
            e_new = g.add_edge(v_win, v_term)
            g.ep["layer"][e_new] = 0
            g.ep["weight"][e_new] = peso

        # caso layer = 1 → transformar em termo como contexto
        elif layer == 1 and {tipo1, tipo2} == {1, 3}:
            v_term, v_win = (v1, v2) if tipo1 == 1 else (v2, v1)
            nome_ctx = g_jan_term.vp["name"][v_term]
            chave_ctx = f"{nome_ctx}<{int(v_term)}>"
            if chave_ctx not in ctx_map:
                v_ctx = g.add_vertex()
                g.vp["name"][v_ctx] = f"<<{nome_ctx}>>"
                g.vp["tipo"][v_ctx] = 4
                g.vp["posicao"][v_ctx] = [-15, cont_y]
                g.vp["color"][v_ctx] = [1, 0.6, 0, 1]
                g.vp["size"][v_ctx] = 10
                cont_y += 1
                ctx_map[chave_ctx] = v_ctx
            else:
                v_ctx = ctx_map[chave_ctx]

            v_win_new = win_map[int(v_win)]
            e_new = g.add_edge(v_ctx, v_win_new)
            g.ep["layer"][e_new] = 0  # agora é camada 0 (não mais 1)
            g.ep["weight"][e_new] = peso

    return g


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
