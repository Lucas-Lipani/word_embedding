from graph_tool.all import minimize_blockmodel_dl, LayeredBlockState


def sbm(g, layered=False):
    if layered == False:
        state = minimize_blockmodel_dl(
            g,
            # state=LayeredBlockState,  # modelo adequado a camadas
            state_args=dict(
                eweight=g.ep["weight"],  # (opcional) multiplicidade da aresta
                pclabel=g.vp["tipo"],  # mantém janelas e termos em grupos separados
            ),
        )
    else:
        state = LayeredBlockState(
            g,
            state=LayeredBlockState,
            state_args={
                "ec": g.ep["layer"],
                "layers": True,
                "eweight": g.ep["weight"],
                "pclabel": g.vp["tipo"],
            },
        )

    return state
