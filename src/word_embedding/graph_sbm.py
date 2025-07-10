from graph_tool.all import minimize_blockmodel_dl, LayeredBlockState

def sbm(g, layered=False, n_blocks=None):
    if layered is False:
        state = minimize_blockmodel_dl(
            g,
            # state=LayeredBlockState,  # modelo adequado a camadas
            state_args=dict(
                eweight=g.ep["weight"],  # (opcional) multiplicidade da aresta
                pclabel=g.vp["tipo"],  # mantém janelas e termos em grupos separados
                B=n_blocks,  # número fixo de blocos (opcional)
            ),
            multilevel_mcmc_args=dict(
                B_min=n_blocks,
                B_max=n_blocks,
            ),
        )
    else:
        state_args = {
            "ec": g.ep["layer"],
            "layers": True,
            "eweight": g.ep["weight"],
            "pclabel": g.vp["tipo"],
        }
        if n_blocks is not None:
            state_args["B"] = n_blocks
        state = LayeredBlockState(
            g,
            state=LayeredBlockState,
            state_args=state_args,
        )
        
    print("=== sbm ===")
    print(state)

    return state
