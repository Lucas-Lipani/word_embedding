from graph_tool.all import Graph
from tqdm import tqdm
from collections import Counter
from pathlib import Path

from . import graph_build


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

    # >>> NOVO: doc_id disponível para qualquer grafo que precise
    doc_id_prop = g.new_vertex_property("string")

    g.vp["amount"] = amount_prop
    g.ep["layer"] = layer_prop
    g.vp["size"] = size_prop
    g.vp["color"] = color_prop
    g.vp["name"] = name_prop
    g.vp["tipo"] = tipo_prop
    g.vp["short_term"] = short_term_prop
    g.vp["posicao"] = posicao_prop
    g.vp["doc_id"] = doc_id_prop  # <<< NOVO
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
        # abstract = row["abstract"]
        abstract = (
            "Janela de teste para analisar se está fazendo tudo certo, caso "
            "esteja tudo certo, irei analisar o próximo."
        )

        print("Abstract")
        print(abstract)

        # ───── vértice Documento ─────
        v_doc = g.add_vertex()
        g.vp["name"][v_doc], g.vp["tipo"][v_doc] = doc_id, 0
        g.vp["posicao"][v_doc] = [-15, doc_y]
        doc_y += 1
        g.vp["size"][v_doc], g.vp["color"][v_doc] = 20, [1, 0, 0, 1]
        g.vp["doc_id"][v_doc] = doc_id  # útil caso precise
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

            # camada 0: Jan → Termo central
            e0 = g.edge(v_win, v_term)
            if e0 is None:
                e0 = g.add_edge(v_win, v_term)
                g.ep["weight"][e0] = 1
                g.ep["layer"][e0] = 0
            else:
                g.ep["weight"][e0] += 1

            # camada 1: Jan → cada termo de contexto
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

    return g


def build_window_graph_and_sliding(df, nlp, w, save_visualizations=False):
    """
    Constrói dois grafos simultaneamente a partir do corpus:
    - g_full: grafo DOCUMENTO–JANELA–TERMO (com camadas)
    - g_slide: grafo Janela deslizante (tipo 5) → Termo (tipo 1)

    A tokenização é feita apenas uma vez por documento.

    :param save_visualizations: Se True, salva PDFs dos grafos construídos
    """
    g_full = initialize_graph()
    g_slide = initialize_graph()
    g_full.vp["termos"] = g_full.new_vertex_property("object")
    g_slide.vp["termos"] = g_slide.new_vertex_property("object")

    doc_y = term_y = win_y = slide_y = slide_term_y = 0
    term_vertex_full = {}
    term_vertex_slide = {}
    doc_vertex = {}
    window_vertex_full = {}
    window_vertex_slide = {}

    for idx, row in tqdm(
        df.iterrows(), desc="Processando documentos", total=len(df)
    ):
        doc_id = str(idx)
        # abstract = row["abstract"]
        # abstract = (
        #     "Janela de teste para analisar se está fazendo tudo certo, caso "
        #     "esteja tudo certo, irei analisar o próximo."
        # )

        # doc = nlp(abstract)
        # tokens = [
        #     t.text.lower().strip() for t in doc if not t.is_stop and not t.is_punct
        # ]
        tokens = row["tokens"]

        # Tamanho local da janela
        w_local = len(tokens) if w == "full" else int(w)

        # ───── Documento no g_full ─────
        v_doc = g_full.add_vertex()
        g_full.vp["name"][v_doc], g_full.vp["tipo"][v_doc] = doc_id, 0
        g_full.vp["posicao"][v_doc] = [-15, doc_y]
        g_full.vp["size"][v_doc], g_full.vp["color"][v_doc] = 20, [1, 0, 0, 1]
        g_full.vp["doc_id"][v_doc] = doc_id
        doc_y += 1
        doc_vertex[doc_id] = v_doc

        for i, term_central in enumerate(tokens):
            # --------- g_full: JANELA (tipo 3) → termo central/contexto ---------
            start, end = max(0, i - w_local), min(len(tokens), i + w_local + 1)
            win_tokens = tokens[start:i] + tokens[i + 1 : end]
            win_key = (frozenset(win_tokens), term_central)

            if win_key not in window_vertex_full:
                v_win = g_full.add_vertex()
                g_full.vp["name"][v_win] = " ".join(win_tokens)
                g_full.vp["tipo"][v_win] = 3
                g_full.vp["termos"][v_win] = win_tokens
                g_full.vp["posicao"][v_win] = [0, win_y]
                g_full.vp["size"][v_win] = 15
                g_full.vp["color"][v_win] = [0.6, 0.6, 0.6, 1]
                window_vertex_full[win_key] = v_win
                win_y += 2
            else:
                v_win = window_vertex_full[win_key]

            if g_full.edge(v_doc, v_win) is None:
                g_full.add_edge(v_doc, v_win)

            if term_central not in term_vertex_full:
                v_term = g_full.add_vertex()
                g_full.vp["name"][v_term] = term_central
                g_full.vp["tipo"][v_term] = 1
                g_full.vp["posicao"][v_term] = [15, term_y]
                g_full.vp["size"][v_term] = 10
                g_full.vp["color"][v_term] = [0, 0, 1, 1]
                g_full.vp["amount"][v_term] = 1
                term_vertex_full[term_central] = v_term
                term_y += 1
            else:
                v_term = term_vertex_full[term_central]
                g_full.vp["amount"][v_term] += 1

            e0 = g_full.edge(v_win, v_term)
            if e0 is None:
                e0 = g_full.add_edge(v_win, v_term)
                g_full.ep["weight"][e0] = 1
                g_full.ep["layer"][e0] = 0
            else:
                g_full.ep["weight"][e0] += 1

            for tok in win_tokens:
                if tok not in term_vertex_full:
                    v_tok = g_full.add_vertex()
                    g_full.vp["name"][v_tok] = tok
                    g_full.vp["tipo"][v_tok] = 1
                    g_full.vp["posicao"][v_tok] = [15, term_y]
                    g_full.vp["size"][v_tok] = 10
                    g_full.vp["color"][v_tok] = [0, 0, 1, 1]
                    g_full.vp["amount"][v_tok] = 1
                    term_vertex_full[tok] = v_tok
                    term_y += 1
                else:
                    v_tok = term_vertex_full[tok]
                    g_full.vp["amount"][v_tok] += 1

                e1 = g_full.edge(v_win, v_tok)
                if e1 is None:
                    e1 = g_full.add_edge(v_win, v_tok)
                    g_full.ep["weight"][e1] = 1
                    g_full.ep["layer"][e1] = 1
                else:
                    g_full.ep["weight"][e1] += 1

        # --------- g_slide: janelas deslizantes por SEQUÊNCIA (ordem preservada), fundidas GLOBALMENTE ---------
        w_local = len(tokens) if w == "full" else int(w)

        # garanta as props (faça isso uma única vez; aqui já funciona porque é antes de criar os vértices da janela)
        if "occurs_total" not in g_slide.vp:
            g_slide.vp["occurs_total"] = g_slide.new_vertex_property("int")
        if "docs" not in g_slide.vp:
            g_slide.vp["docs"] = g_slide.new_vertex_property(
                "object"
            )  # set de doc_ids
        if "occurs_by_doc" not in g_slide.vp:
            g_slide.vp["occurs_by_doc"] = g_slide.new_vertex_property(
                "object"
            )  # dict {doc_id: cont}

        # desliza com passo 1 e tamanho fixo
        for start in range(0, len(tokens) - w_local + 1):
            end = start + w_local
            seq = tuple(tokens[start:end])  # ordem preservada
            seq_key = seq  # <— CHAVE GLOBAL (sem doc_id!)

            v_slide = window_vertex_slide.get(seq_key)
            created = False
            if v_slide is None:
                # cria UMA VEZ para esta sequência (globalmente)
                v_slide = g_slide.add_vertex()
                g_slide.vp["name"][v_slide] = " ".join(seq)
                g_slide.vp["tipo"][v_slide] = 5
                g_slide.vp["posicao"][v_slide] = [0, slide_y]
                g_slide.vp["size"][v_slide] = 12
                g_slide.vp["color"][v_slide] = [0.6, 0.6, 0.0, 1]
                g_slide.vp["termos"][v_slide] = list(seq)
                g_slide.vp["occurs_total"][v_slide] = 0
                g_slide.vp["docs"][v_slide] = set()
                g_slide.vp["occurs_by_doc"][v_slide] = {}
                window_vertex_slide[seq_key] = v_slide
                slide_y += 1
                created = True

            # atualiza contadores globais/por-doc
            g_slide.vp["occurs_total"][v_slide] += 1
            docs_set = g_slide.vp["docs"][v_slide]
            docs_set.add(doc_id)
            g_slide.vp["docs"][v_slide] = docs_set  # reatribui por segurança

            ob = g_slide.vp["occurs_by_doc"][v_slide]
            ob[doc_id] = ob.get(doc_id, 0) + 1
            g_slide.vp["occurs_by_doc"][v_slide] = ob

            # acumula pesos das arestas janela→termo
            freq = Counter(seq)
            for tok, c in freq.items():
                v_tok = term_vertex_slide.get(tok)
                if v_tok is None:
                    v_tok = g_slide.add_vertex()
                    g_slide.vp["name"][v_tok] = tok
                    g_slide.vp["tipo"][v_tok] = 1
                    g_slide.vp["posicao"][v_tok] = [15, slide_term_y]
                    g_slide.vp["size"][v_tok] = 10
                    g_slide.vp["color"][v_tok] = [0, 0, 1, 1]
                    term_vertex_slide[tok] = v_tok
                    slide_term_y += 1
                e = g_slide.edge(v_slide, v_tok)
                if e is None:
                    e = g_slide.add_edge(v_slide, v_tok)
                    g_slide.ep["layer"][e] = 0
                    g_slide.ep["weight"][
                        e
                    ] = c  # primeira vez dessa janela global
                else:
                    g_slide.ep["weight"][
                        e
                    ] += c  # janela repetiu (outro start/mesmo doc ou outro doc)

    if save_visualizations:
        save_graph_visualization(g_full, f"01_Document-Window-Term_window{w}")
        save_graph_visualization(
            g_slide, f"02_Document-SlideWindow-Term_window{w}"
        )

    return g_full, g_slide


def save_graph_visualization(g, filename: str, layout=None):
    """
    Salva visualização do grafo em PDF com nome descritivo.

    :param g: Grafo a visualizar
    :param filename: Nome do arquivo (sem extensão, será .pdf)
    :param layout: Layout (pos) para os vértices. Se None, usa sfdp_layout.
    """
    from graph_tool.all import sfdp_layout, graph_draw

    out_dir = _ensure_output_dir()
    output_path = out_dir / f"{filename}.pdf"

    if layout is None:
        layout = sfdp_layout(g)

    graph_draw(
        g,
        pos=g.vp["posicao"],
        vertex_text=g.vp["name"],
        vertex_text_position=-2,
        vertex_text_color="black",
        vertex_font_size=10,
        vertex_fill_color=g.vp["color"],
        vertex_size=g.vp["size"],
        output=str(output_path),
    )

    print(f"[GRAPH] Grafo salvo em: {output_path}")
    return output_path


def _ensure_output_dir():
    """
    Cria o diretório de output para PDFs se não existir.
    """
    out_dir = Path("../outputs/graphs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


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
        g_win_term.vp[prop] = g_win_term.new_vertex_property(
            g.vp[prop].value_type()
        )
    for prop in g.ep.keys():
        g_win_term.ep[prop] = g_win_term.new_edge_property(
            g.ep[prop].value_type()
        )

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
        g_doc_term.vp[prop] = g_doc_term.new_vertex_property(
            g.vp[prop].value_type()
        )
    for prop in g.ep.keys():
        g_doc_term.ep[prop] = g_doc_term.new_edge_property(
            g.ep[prop].value_type()
        )

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
