from graph_tool.all import minimize_blockmodel_dl, LayeredBlockState

def sbm(g, layered=False, n_blocks=None):
    if not layered:
        # Define state_args básicos
        state_args = dict(
            eweight=g.ep["weight"],
            pclabel=g.vp["tipo"]
        )

        # Inclui número fixo de blocos, se fornecido
        if n_blocks is not None:
            state_args["B"] = n_blocks
            mcmc_args = dict(B_min=n_blocks, B_max=n_blocks)
        else:
            mcmc_args = {}

        # Chamada do modelo SBM
        state = minimize_blockmodel_dl(
            g,
            state_args=state_args,
            multilevel_mcmc_args=mcmc_args
        )
    else:
        state_args = {
            "ec": g.ep["layer"],
            "layers": True,
            "eweight": g.ep["weight"],
            "pclabel": g.vp["tipo"]
        }

        if n_blocks is not None:
            state_args["B"] = n_blocks

        state = LayeredBlockState(
            g,
            state=LayeredBlockState,
            state_args=state_args
        )

    print("=== sbm ===")
    print(state)

    return state
