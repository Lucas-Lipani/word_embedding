from pathlib import Path
import pandas as pd
import argparse
import sys
import json
import numpy as np
from graph_tool.all import (
    variation_information,
    partition_overlap,
    mutual_information,
    reduced_mutual_information,
)
from sklearn.metrics import adjusted_rand_score
from . import analysis_manager


def _compare_metrics(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """Compara duas séries de rótulos de partição."""
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"As duas séries de rótulos devem ter o mesmo comprimento. "
            f"Recebido: {len(labels_a)} vs {len(labels_b)}"
        )
    return {
        "vi": variation_information(labels_a, labels_b, norm=False),
        "nvi": variation_information(labels_a, labels_b, norm=True),
        "po": partition_overlap(labels_a, labels_b, norm=False),
        "npo": partition_overlap(labels_a, labels_b, norm=True),
        "mi": mutual_information(labels_a, labels_b, norm=False, adjusted=False),
        "ami": mutual_information(labels_a, labels_b, norm=False, adjusted=True),
        "nmi": mutual_information(labels_a, labels_b, norm=True, adjusted=False),
        "anmi": mutual_information(labels_a, labels_b, norm=True, adjusted=True),
        "ari": adjusted_rand_score(labels_a, labels_b),
        "rmi": reduced_mutual_information(labels_a, labels_b, norm=False),
        "nrmi": reduced_mutual_information(labels_a, labels_b, norm=True),
    }


def find_all_configs_by_corpus(base_conf_dir: Path, seed: int, n_samples: int) -> dict:
    """
    Encontra TODAS as configs que compartilham seed + n_samples.
    
    :return: dict {config_idx: {"model": "sbm"|"w2v+kmeans", "windows": [...]}}
    """
    configs_found = {}
    
    for config_dir in sorted(base_conf_dir.glob("????")):
        config_file = config_dir / "config.json"
        if not config_file.exists():
            continue
        
        try:
            with open(config_file, "r") as f:
                cfg = json.load(f)
            
            cfg_seed = cfg.get("corpus", {}).get("seed")
            cfg_samples = cfg.get("corpus", {}).get("number_of_documents")
            cfg_model_kind = cfg.get("model", {}).get("kind")
            cfg_window = cfg.get("graph", {}).get("window_size")
            
            # Filtrar por seed e samples
            if cfg_seed != seed or cfg_samples != n_samples:
                continue
            
            config_idx = int(config_dir.name)
            
            if config_idx not in configs_found:
                configs_found[config_idx] = {
                    "model": cfg_model_kind,
                    "windows": [],
                    "config_dir": config_dir,
                }
            
            configs_found[config_idx]["windows"].append(str(cfg_window))
            
            print(f"  [FOUND] Config {config_idx:04d}: model={cfg_model_kind}, window={cfg_window}")
        
        except Exception as e:
            print(f"  [WARN] Erro ao ler {config_file}: {e}")
    
    return configs_found


def load_partition_data(config_dir: Path) -> pd.DataFrame:
    """Carrega TODOS os parquets de uma config."""
    run_dirs = sorted(config_dir.glob("run/????"))
    dfs = []
    
    for run_dir in run_dirs:
        partition_file = run_dir / "partition.parquet"
        if not partition_file.exists():
            continue
        
        try:
            run_idx = int(run_dir.name)
        except ValueError:
            continue
        
        df = pd.read_parquet(partition_file)
        df["config"] = int(config_dir.name)
        df["run"] = run_idx
        dfs.append(df)
    
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)


def compute_global_analysis(
    seed: int,
    n_samples: int,
    base_conf_dir: Path = Path("../outputs/conf"),
    base_analyses_dir: Path = Path("../outputs/analyses"),
):
    """
    Comparações GLOBAIS entre TODAS as configs com seed + n_samples.
    Executa: sbm_vs_sbm, sbm_vs_w2v, w2v_vs_w2v.
    """
    
    print(f"\n[ANALYSIS] Procurando TODAS as configs:")
    print(f"  seed={seed}, samples={n_samples}")
    
    # Encontrar todas as configs
    configs_found = find_all_configs_by_corpus(base_conf_dir, seed, n_samples)
    
    if not configs_found:
        print(f"[ERROR] Nenhuma config encontrada", file=sys.stderr)
        return False
    
    print(f"\n[FOUND] {len(configs_found)} configs encontradas:")
    for idx in sorted(configs_found.keys()):
        cfg = configs_found[idx]
        print(f"  Config {idx:04d}: {cfg['model']} | windows: {cfg['windows']}")
    
    # Agrupar configs por modelo
    sbm_configs = {idx: cfg for idx, cfg in configs_found.items() if cfg["model"] == "sbm"}
    w2v_configs = {idx: cfg for idx, cfg in configs_found.items() if cfg["model"] == "w2v+kmeans"}
    
    print(f"\n  SBM configs: {list(sbm_configs.keys())}")
    print(f"  W2V configs: {list(w2v_configs.keys())}")
    
    # Carregar dados de TODAS as configs
    all_data = {}
    for config_idx, cfg in configs_found.items():
        data = load_partition_data(cfg["config_dir"])
        if data is not None:
            all_data[config_idx] = data
        else:
            print(f"    [WARN] Nenhum parquet em config {config_idx:04d}")
    
    if not all_data:
        print(f"[ERROR] Sem dados para análise", file=sys.stderr)
        return False
    
    # Combinar TODOS os dados em um único DataFrame
    data = pd.concat(all_data.values(), ignore_index=True)
    data["window"] = data["window"].astype(str)
    
    # ===== COMPARAÇÕES GLOBAIS =====
    comparisons = [
        ("sbm", "sbm", "sbm_vs_sbm"),
        ("sbm", "w2v+kmeans", "sbm_vs_w2v"),
        ("w2v+kmeans", "w2v+kmeans", "w2v_vs_w2v"),
    ]
    
    all_success = True
    
    for model_x, model_y, comparison_name in comparisons:
        print(f"\n[COMPARE] {comparison_name}")
        
        # >>> CORRIGIDO: determinar quais configs usar baseado no modelo
        if model_x == "sbm":
            valid_configs_x = list(sbm_configs.keys())
        else:
            valid_configs_x = list(w2v_configs.keys())
        
        if model_y == "sbm":
            valid_configs_y = list(sbm_configs.keys())
        else:
            valid_configs_y = list(w2v_configs.keys())
        
        print(f"    Valid configs X ({model_x}): {valid_configs_x}")
        print(f"    Valid configs Y ({model_y}): {valid_configs_y}")
        
        # >>> NOVO: carregar dados SEPARADAMENTE por config
        # Cada config agora contém APENAS seu modelo específico
        dfs_x = []
        for config_idx in valid_configs_x:
            cfg = configs_found[config_idx]
            if all_data.get(config_idx) is not None:
                df_tmp = all_data[config_idx].copy()
                # >>> NÃO precisa filtrar por modelo agora, pois parquet já contém apenas o modelo correto
                if not df_tmp.empty:
                    dfs_x.append(df_tmp)
        
        dfs_y = []
        for config_idx in valid_configs_y:
            cfg = configs_found[config_idx]
            if all_data.get(config_idx) is not None:
                df_tmp = all_data[config_idx].copy()
                # >>> NÃO precisa filtrar por modelo agora
                if not df_tmp.empty:
                    dfs_y.append(df_tmp)
        
        if not dfs_x or not dfs_y:
            print(f"  [SKIP] Sem dados para {model_x} ou {model_y}")
            all_success = False
            continue
        
        data_x = pd.concat(dfs_x, ignore_index=True)
        data_y = pd.concat(dfs_y, ignore_index=True)
        
        data_x["window"] = data_x["window"].astype(str)
        data_y["window"] = data_y["window"].astype(str)
        
        configs_x = sorted(data_x["config"].unique())
        configs_y = sorted(data_y["config"].unique())
        windows_x = sorted(data_x["window"].unique())
        windows_y = sorted(data_y["window"].unique())
        runs_x = sorted(data_x["run"].unique())
        runs_y = sorted(data_y["run"].unique())
        
        print(f"    Configs X ({model_x}): {list(configs_x)}")
        print(f"    Configs Y ({model_y}): {list(configs_y)}")
        print(f"    Windows X: {list(windows_x)}")
        print(f"    Windows Y: {list(windows_y)}")
        
        rows = []
        
        # >>> NOVO: comparar TODAS as combinações de configs
        for cx in configs_x:
            for cy in configs_y:
                # Evitar auto-comparação exata APENAS se mesmo modelo E mesma config
                # if model_x == model_y and cx == cy:
                #     continue
                
                for wx in windows_x:
                    for wy in windows_y:
                        for rx in runs_x:
                            for ry in runs_y:
                                # Skip only exact self-comparison: same config AND same run
                                if model_x == model_y and cx == cy and rx == ry:
                                    continue
                                df_rx = data_x[
                                    (data_x["config"] == cx)
                                    & (data_x["window"] == wx)
                                    & (data_x["run"] == rx)
                                ]
                                df_ry = data_y[
                                    (data_y["config"] == cy)
                                    & (data_y["window"] == wy)
                                    & (data_y["run"] == ry)
                                ]
                                
                                if df_rx.empty or df_ry.empty:
                                    continue
                                
                                # APENAS termos (tipo==1)
                                df_rx_terms = df_rx[
                                    (df_rx["tipo"] == 1) & (df_rx["term"].notna())
                                ].set_index("term")
                                df_ry_terms = df_ry[
                                    (df_ry["tipo"] == 1) & (df_ry["term"].notna())
                                ].set_index("term")
                                
                                common = df_rx_terms.index.intersection(
                                    df_ry_terms.index
                                )
                                if len(common) == 0:
                                    continue
                                
                                labels_x = df_rx_terms.loc[common]["label"].values
                                labels_y = df_ry_terms.loc[common]["label"].values
                                
                                try:
                                    metrics = _compare_metrics(labels_x, labels_y)
                                    row = {
                                        "config_x": cx,
                                        "config_y": cy,
                                        "run_x": rx,
                                        "run_y": ry,
                                        "window_x": wx,
                                        "window_y": wy,
                                    }
                                    row.update(metrics)
                                    rows.append(row)
                                except ValueError:
                                    continue
        
        if not rows:
            print(f"  [WARN] Sem comparações válidas para {model_x}_vs_{model_y}")
            all_success = False
            continue
        
        df_result = pd.DataFrame(rows)
        
        # >>> CORRIGIDO: usar TODAS as configs relevantes
        all_config_indices = sorted(set(list(configs_x) + list(configs_y)))
        
        ana_mgr = analysis_manager.AnalysisManager(base_analyses_dir)
        analysis_dir, analysis_idx = ana_mgr.find_or_create_analysis_dir(
            all_config_indices, comparison_name
        )
        
        # Salvar resultados
        results_parquet = analysis_dir / "results.parquet"
        df_result.to_parquet(results_parquet, engine="pyarrow")
        print(f"  [SAVED] results.parquet: {results_parquet}")
        
        ana_mgr.save_analysis_config(
            analysis_dir,
            all_config_indices,
            comparison_name,
            corpus_seed=seed,
            n_samples=n_samples,
            graph_type=None,  # Não importa, é apenas referência
        )
        
        ana_mgr.save_analysis_results(
            analysis_dir,
            df_result,
            comparison_name,
        )
        
        print(f"  [ANALYSIS {analysis_idx:04d}] {comparison_name}")
        print(f"    Configs: {all_config_indices}")
        print(f"    Seed: {seed}, Samples: {n_samples}")
        print(f"    Total de comparações: {len(df_result)}")
    
    return all_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Análises GLOBAIS entre TODAS as configs (seed + samples)."
    )
    parser.add_argument(
        "--seed", "-s", type=int, required=True, help="Seed do corpus"
    )
    parser.add_argument(
        "--samples", type=int, required=True, help="Número de amostras"
    )
    args = parser.parse_args()
    
    all_success = compute_global_analysis(args.seed, args.samples)
    
    if all_success:
        print(f"\n✓ Análises concluídas com sucesso!")
        sys.exit(0)
    else:
        print(f"\n✗ Alguns erros ocorreram")
        sys.exit(1)
