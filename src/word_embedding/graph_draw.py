from graph_tool.all import (
    # prop_to_size,
    # sfdp_layout,
    graph_draw,
)
import os


def draw_base_graphs(g, g_doc_jan, g_doc_term, g_con_jan_term, window):
    window = str(window)
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../outputs/window")
    )
    os.makedirs(base_dir, exist_ok=True)

    graph_draw(
        g,
        pos=g.vp["posicao"],
        vertex_text=g.vp["name"],
        vertex_text_position=-2,
        vertex_text_color="black",
        vertex_font_size=10,
        vertex_fill_color=g.vp["color"],
        output=os.path.join(base_dir, f"{window}_graph_d-j-t.pdf"),
    )

    graph_draw(
        g_doc_jan,
        pos=g_doc_jan.vp["posicao"],
        vertex_text=g_doc_jan.vp["name"],
        vertex_text_position=-2,
        vertex_text_color="black",
        vertex_font_size=10,
        vertex_fill_color=g_doc_jan.vp["color"],
        output=os.path.join(base_dir, f"window{window}_graph_d-j.pdf"),
    )

    graph_draw(
        g_doc_term,
        pos=g_doc_term.vp["posicao"],
        vertex_text=g_doc_term.vp["name"],
        vertex_text_position=-2,
        vertex_text_color="black",
        vertex_font_size=10,
        vertex_fill_color=g_doc_term.vp["color"],
        output=os.path.join(base_dir, f"window{window}_graph_d-t.pdf"),
    )

    graph_draw(
        g_con_jan_term,
        pos=g_con_jan_term.vp["posicao"],
        vertex_text=g_con_jan_term.vp["name"],
        vertex_text_position=-2,
        vertex_text_color="black",
        vertex_font_size=10,
        vertex_fill_color=g_con_jan_term.vp["color"],
        output=os.path.join(base_dir, f"window{window}_graph_c-j-t.pdf"),
    )
