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
    sbm_entropy: float | None = None,
    # >>> Informações detalhadas do grafo e SBM
    vertices_pre_sbm: Dict[int, int] = None,
    blocks_post_sbm: Dict[int, int] = None,
    term_blocks_count: int = None,
    window_blocks_count: int = None,
    # >>> Informações do W2V
    w2v_n_clusters: int = None,
    w2v_sg: int = None,
    w2v_window: int = None,
    w2v_vector_size: int = None,
):
    """
    Salva partições na nova estrutura: conf/NNNN/run/RRRR/partition.parquet
    Com informações detalhadas sobre estrutura do grafo, blocos do SBM e W2V.
    """
    cfg_mgr = config_manager.ConfigManager(base_conf_dir)

    # Encontrar ou criar config_dir
    config_dir, config_idx, was_reused = cfg_mgr.find_or_create_config_dir(
        n_samples=n_samples,
        seed=seed,
        graph_type=graph_type,
        nested=nested,
        n_blocks=n_blocks,
    )

    # Salvar config.json
    cfg_mgr.save_config(
        config_dir=config_dir,
        n_samples=n_samples,
        seed=seed,
        graph_type=graph_type,
        nested=nested,
        n_blocks=n_blocks,
    )

    # Calcular o próximo run_idx disponível
    run_dir_base = config_dir / "run"
    run_dir_base.mkdir(parents=True, exist_ok=True)

    existing_runs = sorted(run_dir_base.glob("????"))
    next_run_idx = (
        max([int(d.name) for d in existing_runs], default=0) + 1
        if existing_runs
        else 1
    )

    print(f"[RUN_IDX] Próximo run_idx disponível: {next_run_idx:04d}")

    run_dir = run_dir_base / f"{next_run_idx:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Salvar partition.parquet
    partition_file = run_dir / "partition.parquet"
    partitions_df.to_parquet(partition_file, engine="pyarrow")

    # Salvar parameters.json COM TODAS AS INFORMAÇÕES (SBM + W2V)
    cfg_mgr.save_run_parameters(
        config_dir=config_dir,
        run_idx=next_run_idx,
        sbm_entropy=sbm_entropy,
        vertices_pre_sbm=vertices_pre_sbm,
        blocks_post_sbm=blocks_post_sbm,
        term_blocks_count=term_blocks_count,
        window_blocks_count=window_blocks_count,
        w2v_n_clusters=w2v_n_clusters,
        w2v_sg=w2v_sg,
        w2v_window=w2v_window,
        w2v_vector_size=w2v_vector_size,
    )

    print(f"[SAVED] Partição salva: {partition_file}")

    return config_idx, next_run_idx, partition_file


def load_partitions(config_dir: Path, run_idx: int) -> pd.DataFrame:
    """
    Carrega partições de conf/NNNN/run/RRRR/partition.parquet

    :param config_dir: Pasta config/NNNN
    :param run_idx: Índice da execução
    :return: DataFrame com partições
    """
    partition_file = (
        config_dir / "run" / f"{run_idx:04d}" / "partition.parquet"
    )

    if not partition_file.exists():
        raise FileNotFoundError(f"Partição não encontrada: {partition_file}")

    return pd.read_parquet(partition_file, engine="pyarrow")
