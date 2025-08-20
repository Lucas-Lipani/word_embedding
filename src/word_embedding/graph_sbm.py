from graph_tool.all import minimize_blockmodel_dl, LayeredBlockState, BlockState
import numpy as np


def sbm(g, layered=False, n_blocks=None):
    if not layered:
        # Define state_args básicos
        state_args = dict(eweight=g.ep["weight"], pclabel=g.vp["tipo"])

        # Inclui número fixo de blocos, se fornecido
        if n_blocks is not None:
            state_args["B"] = n_blocks
            mcmc_args = dict(B_min=n_blocks, B_max=n_blocks)
        else:
            mcmc_args = {}

        # Chamada do modelo SBM
        state = minimize_blockmodel_dl(
            g, state_args=state_args, multilevel_mcmc_args=mcmc_args
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

        state = LayeredBlockState(g, state=LayeredBlockState, state_args=state_args)

    print("=== sbm ===")
    print(state)

    return state


def sbm_with_fixed_term_blocks(g, n_term_blocks, init_method="random"):
    """
    Roda SBM no grafo g, fixando que os vértices tipo TERMO (tipo=1) fiquem em
    exatamente n_term_blocks blocos. Os demais vértices permanecem livres.
    """

    b_init = g.new_vertex_property("int")
    clabel = g.new_vertex_property("int")

    np.random.seed(42)
    free_block = 0
    max_free_blocks = 1000
    offset = 10_000  # separa blocos livres dos blocos fixos

    tipo_counts = {}
    total = 0

    for v in g.vertices():
        tipo = int(g.vp["tipo"][v])
        tipo_counts[tipo] = tipo_counts.get(tipo, 0) + 1
        total += 1

        if tipo == 1:  # TERMO
            if init_method == "random":
                bloco = np.random.randint(0, n_term_blocks)
            else:
                bloco = hash(g.vp["name"][v]) % n_term_blocks
            b_init[v] = bloco
            clabel[v] = bloco
        else:
            bloco_livre = offset + (free_block % max_free_blocks)
            b_init[v] = bloco_livre
            clabel[v] = -1
            free_block += 1

    print(f"[DEBUG] Total de vértices: {total}")
    print(f"[DEBUG] Contagem por tipo: {tipo_counts}")
    print(f"[DEBUG] Total de blocos livres usados: {free_block}")
    print(f"[DEBUG] b_init size: {b_init.a.shape[0]}, clabel size: {clabel.a.shape[0]}")

    assert b_init.a.shape[0] == g.num_vertices()
    assert clabel.a.shape[0] == g.num_vertices()

    # VERIFICAÇÕES ADICIONAIS
    for v in g.vertices():
        if clabel[v] != -1 and v.out_degree() + v.in_degree() == 0:
            print(
                f"[ERRO] Vértice fixado sem arestas: {v}, tipo={int(g.vp['tipo'][v])}, name={g.vp['name'][v]}"
            )
        if clabel[v] != -1 and clabel[v] != b_init[v]:
            print(
                f"[ERRO] clabel ≠ b_init para v {v}: clabel={clabel[v]}, b={b_init[v]}"
            )

    for e in g.edges():
        if e.source() == e.target():
            print(f"[ERRO] Loop detectado: {e}")
        if g.ep["weight"][e] <= 0:
            print(f"[ERRO] Peso inválido: {e}, peso={g.ep['weight'][e]}")

    # Opcional: exporta grafo p/ debug externo
    try:
        g.save("debug_graph.gt")
        print("[DEBUG] Grafo exportado para 'debug_graph.gt'")
    except Exception as e:
        print("[WARN] Falha ao exportar grafo:", e)

    # Execução principal
    try:
        state = BlockState(
            g,
            b=b_init,
            clabel=clabel,
            eweight=g.ep["weight"],
            # REMOVIDO: pclabel=g.vp["tipo"]
        )
        print(f"[SBM] Termos fixados em {n_term_blocks} blocos. Rodando MCMC...")
        state.mcmc_anneal(beta_range=(1, 20), niter=5000)
        print("[OK] MCMC finalizado com sucesso.")
    except Exception as e:
        print("[ERRO] Durante execução do SBM com blocos fixos:", e)
        raise

    return state
