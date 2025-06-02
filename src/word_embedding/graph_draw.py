from graph_tool.all import (
    # prop_to_size,
    # LayeredBlockState,
    graph_draw,
    # sfdp_layout,
    graph_draw,
)


def draw_base_graphs(g, g_doc_jan, g_doc_term, g_con_jan_term, window):

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
        output="../../outputs/window"
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
        output="../../outputs/window/window"
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
        output="../../outputs/window/window"
        + window
        + "_graph_d-t.pdf",  # Salva a visualização em PDF
    )

    # Salva o grafo original DOCUMENTO - TERMOS
    graph_draw(
        g_con_jan_term,
        # pos=sfdp_layout(g_con_jan_term),           # Layout para posicionar os nós
        pos=g_con_jan_term.vp["posicao"],
        vertex_text=g_con_jan_term.vp[
            "name"
        ],  # Usa o rótulo armazenado na propriedade "name"
        vertex_text_position=-2,
        vertex_text_color="black",
        vertex_font_size=10,  # Tamanho da fonte dos rótulos
        vertex_fill_color=g_con_jan_term.vp["color"],  # Define a cor dos vértices
        output="../../outputs/window/window"
        + window
        + "_graph_c-j-t.pdf",  # Salva a visualização em PDF
    )
