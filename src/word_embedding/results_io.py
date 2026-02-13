# word_embedding/results_io.py
from pathlib import Path
from typing import Dict
import pandas as pd
from . import config_manager


def save_partitions_by_config(
    base_conf_dir: Path,
    n_samples: int,
    seed: int,
    graph_type: str,
    nested: bool,
    n_blocks: int | None,
    run_idx: int,
    partitions_df: pd.DataFrame,
    window_size: int | str,
    sbm_entropy: float | None = None,
    vertices_pre_sbm: Dict[int, int] = None,
    blocks_post_sbm: Dict[int, int] = None,
    term_blocks_count: int = None,
    window_blocks_count: int = None,
    w2v_n_clusters: int = None,
):
    """
    Salva partições na nova estrutura: conf/NNNN/run/RRRR/partition.parquet
    Com config.json (ENTRADA) e results.json (SAÍDA) separados por modelo.

    >>> IMPORTANTE: Salva APENAS dados do modelo específico em cada config <<<
    """
    cfg_mgr = config_manager.ConfigManager(base_conf_dir)

    # Encontrar ou criar 2 configs (SBM + W2V) com mesma assinatura de corpus+window
    config_dir_sbm, config_dir_w2v, config_idx = (
        cfg_mgr.find_or_create_config_dirs(
            n_samples=n_samples,
            seed=seed,
            graph_type=graph_type,
            nested=nested,
            n_blocks=n_blocks,
            window_size=window_size,
        )
    )

    # Salvar config.json para SBM
    cfg_mgr.save_config(
        config_dir=config_dir_sbm,
        model_kind="sbm",
        n_samples=n_samples,
        seed=seed,
        graph_type=graph_type,
        nested=nested,
        n_blocks=n_blocks,
        window_size=window_size,
    )

    # Salvar config.json para W2V
    cfg_mgr.save_config(
        config_dir=config_dir_w2v,
        model_kind="w2v",
        n_samples=n_samples,
        seed=seed,
        graph_type=graph_type,
        nested=nested,
        n_blocks=n_blocks,
        window_size=window_size,
    )

    # Calcular o próximo run_idx disponível (usa SBM como referência)
    run_dir_base_sbm = config_dir_sbm / "run"
    run_dir_base_sbm.mkdir(parents=True, exist_ok=True)

    existing_runs = sorted(run_dir_base_sbm.glob("????"))
    next_run_idx = (
        max([int(d.name) for d in existing_runs], default=0) + 1
        if existing_runs
        else 1
    )

    print(f"[RUN_IDX] Próximo run_idx disponível: {next_run_idx:04d}")

    # Criar diretórios de run para ambas configs
    run_dir_sbm = run_dir_base_sbm / f"{next_run_idx:04d}"
    run_dir_sbm.mkdir(parents=True, exist_ok=True)

    run_dir_w2v = config_dir_w2v / "run" / f"{next_run_idx:04d}"
    run_dir_w2v.mkdir(parents=True, exist_ok=True)

    # >>> CORRIGIDO: Salvar APENAS dados do modelo específico em cada config
    partitions_sbm = partitions_df[partitions_df["model"] == "sbm"].copy()
    partitions_w2v = partitions_df[partitions_df["model"] == "w2v"].copy()

    partition_file_sbm = run_dir_sbm / "partition.parquet"
    partition_file_w2v = run_dir_w2v / "partition.parquet"

    if not partitions_sbm.empty:
        partitions_sbm.to_parquet(partition_file_sbm, engine="pyarrow")
        print(
            f"[SAVED] Partição SBM: {partition_file_sbm} ({len(partitions_sbm)} rows)"
        )
    else:
        print(f"[WARN] Nenhum dado SBM para salvar em {partition_file_sbm}")

    if not partitions_w2v.empty:
        partitions_w2v.to_parquet(partition_file_w2v, engine="pyarrow")
        print(
            f"[SAVED] Partição W2V: {partition_file_w2v} ({len(partitions_w2v)} rows)"
        )
    else:
        print(f"[WARN] Nenhum dado W2V para salvar em {partition_file_w2v}")

    # Salvar results.json para SBM
    cfg_mgr.save_run_results(
        config_dir=config_dir_sbm,
        run_idx=next_run_idx,
        model_kind="sbm",
        sbm_entropy=sbm_entropy,
        vertices_pre_sbm=vertices_pre_sbm,
        partitions_per_type=blocks_post_sbm,
    )

    # Salvar results.json para W2V
    cfg_mgr.save_run_results(
        config_dir=config_dir_w2v,
        run_idx=next_run_idx,
        model_kind="w2v",
        w2v_n_clusters=w2v_n_clusters,
        clusters_per_type={1: w2v_n_clusters} if w2v_n_clusters else None,
    )

    return config_idx, next_run_idx, partition_file_sbm
