from pathlib import Path
import pandas as pd
import argparse
import sys
import json
import numpy as np
from collections import defaultdict
from graph_tool.all import (
    variation_information,
    partition_overlap,
    mutual_information,
    reduced_mutual_information,
)
from sklearn.metrics import adjusted_rand_score
from . import analysis_manager


def _compare_metrics(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """Compare two partition label arrays."""
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Both label arrays must have the same length. "
            f"Got: {len(labels_a)} vs {len(labels_b)}"
        )
    # Cast to int (safer for graph-tool + sklearn)
    a = labels_a.astype(int)
    b = labels_b.astype(int)

    return {
        "vi": float(variation_information(a, b, norm=False)),
        "nvi": float(variation_information(a, b, norm=True)),
        "po": float(partition_overlap(a, b, norm=False)),
        "npo": float(partition_overlap(a, b, norm=True)),
        "mi": float(mutual_information(a, b, norm=False, adjusted=False)),
        "ami": float(mutual_information(a, b, norm=False, adjusted=True)),
        "nmi": float(mutual_information(a, b, norm=True, adjusted=False)),
        "anmi": float(mutual_information(a, b, norm=True, adjusted=True)),
        "ari": float(adjusted_rand_score(a, b)),
        "rmi": float(reduced_mutual_information(a, b, norm=False)),
        "nrmi": float(reduced_mutual_information(a, b, norm=True)),
    }


def find_all_configs_by_corpus(
    base_conf_dir: Path, seed: int, n_samples: int
) -> dict:
    """
    Find ALL configs that share seed + n_samples.

    Returns:
      dict {config_idx: {"model": "sbm"|"w2v+kmeans", "windows": [...], "config_dir": Path}}
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

            print(
                f"  [FOUND] Config {config_idx:04d}: model={cfg_model_kind}, window={cfg_window}"
            )

        except Exception as e:
            print(f"  [WARN] Error reading {config_file}: {e}")

    return configs_found


def load_partition_data(config_dir: Path) -> pd.DataFrame | None:
    """Load ALL parquet partitions from a config directory."""
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


# ----------------------------
# FAST INDEX (core optimization)
# ----------------------------
def build_term_index(df: pd.DataFrame):
    """
    Build fast lookups:
      idx[(config, window, run)] -> pd.Series(label, index=term)
      runs_by[(config, window)]  -> sorted list of runs available

    This avoids filtering df inside deep loops.
    """
    if df is None or df.empty:
        return {}, {}

    required = {"config", "window", "run", "tipo", "term", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    df2 = df[(df["tipo"] == 1) & (df["term"].notna())][
        ["config", "window", "run", "term", "label"]
    ].copy()

    df2["config"] = df2["config"].astype(int)
    df2["run"] = df2["run"].astype(int)
    df2["window"] = df2["window"].astype(str)

    idx = {}
    runs_by = defaultdict(set)

    for (c, w, r), sub in df2.groupby(["config", "window", "run"], sort=False):
        idx[(c, w, r)] = sub.set_index("term")["label"]
        runs_by[(c, w)].add(r)

    runs_by = {k: sorted(v) for k, v in runs_by.items()}
    return idx, runs_by


def compare_cross_all_windows_aligned_runs(
    idx_a: dict,
    runs_by_a: dict,
    configs_a: list[int],
    model_a: str,
    idx_b: dict,
    runs_by_b: dict,
    configs_b: list[int],
    model_b: str,
    min_common_terms: int = 2,
):
    """
    Cross-modal:
      - compare config_a vs config_b
      - compare ALL window pairs (wx, wy)
      - compare aligned runs (same run id when available on both sides)
    """
    rows = []

    windows_by_config_a = defaultdict(set)
    runs_by_config_window_a = defaultdict(set)
    for (c, w), runs in runs_by_a.items():
        windows_by_config_a[c].add(w)
        runs_by_config_window_a[(c, w)] = set(runs)

    windows_by_config_b = defaultdict(set)
    runs_by_config_window_b = defaultdict(set)
    for (c, w), runs in runs_by_b.items():
        windows_by_config_b[c].add(w)
        runs_by_config_window_b[(c, w)] = set(runs)

    for ca in configs_a:
        wa = sorted(windows_by_config_a.get(ca, set()))
        if not wa:
            continue

        for cb in configs_b:
            wb = sorted(windows_by_config_b.get(cb, set()))
            if not wb:
                continue

            for wx in wa:
                runs_ax = runs_by_config_window_a.get((ca, wx), set())
                if not runs_ax:
                    continue

                for wy in wb:
                    runs_by = runs_by_config_window_b.get((cb, wy), set())
                    if not runs_by:
                        continue

                    common_runs = sorted(runs_ax.intersection(runs_by))
                    if not common_runs:
                        continue

                    for r in common_runs:
                        sa = idx_a.get((ca, wx, r))
                        sb = idx_b.get((cb, wy, r))
                        if sa is None or sb is None:
                            continue

                        common = sa.index.intersection(sb.index)
                        if len(common) < min_common_terms:
                            continue

                        la = sa.loc[common].to_numpy()
                        lb = sb.loc[common].to_numpy()

                        try:
                            metrics = _compare_metrics(la, lb)
                        except ValueError:
                            continue

                        row = {
                            "config_x": ca,
                            "config_y": cb,
                            "run_x": r,
                            "run_y": r,
                            "window_x": wx,
                            "window_y": wy,
                            "model_x": model_a,
                            "model_y": model_b,
                            "n_common_terms": int(len(common)),
                        }
                        row.update(metrics)
                        rows.append(row)

    return rows


def compare_intra_windows_full_heatmap(
    idx: dict,
    runs_by: dict,
    configs: list[int],
    model_name: str,
    min_common_terms: int = 2,
):
    """
    Intra-modal FULL heatmap:

    A) Diagonal (w,w):
       - within the same config/window
       - compare run pairs (r1 < r2) to measure run variability

    B) Off-diagonal (wx, wy):
       - compare different configs (cx != cy), each config has its own window
       - compare aligned runs only (same run id) to keep cost low
    """
    rows = []

    # Map config -> windows available (in your case usually 1)
    windows_by_config = defaultdict(set)
    for c, w in runs_by.keys():
        windows_by_config[c].add(w)

    # ---------- A) DIAGONAL: run-pairs inside each config/window ----------
    for c in configs:
        windows = sorted(windows_by_config.get(c, set()))
        for w in windows:
            runs = runs_by.get((c, w), [])
            if len(runs) < 2:
                continue

            for i in range(len(runs)):
                r1 = runs[i]
                s1 = idx.get((c, w, r1))
                if s1 is None:
                    continue
                for j in range(i + 1, len(runs)):
                    r2 = runs[j]
                    s2 = idx.get((c, w, r2))
                    if s2 is None:
                        continue

                    common = s1.index.intersection(s2.index)
                    if len(common) < min_common_terms:
                        continue

                    labels_1 = s1.loc[common].to_numpy()
                    labels_2 = s2.loc[common].to_numpy()

                    try:
                        metrics = _compare_metrics(labels_1, labels_2)
                    except ValueError:
                        continue

                    row = {
                        "config_x": c,
                        "config_y": c,
                        "run_x": r1,
                        "run_y": r2,
                        "window_x": w,
                        "window_y": w,
                        "model_x": model_name,
                        "model_y": model_name,
                        "n_common_terms": int(len(common)),
                    }
                    row.update(metrics)
                    rows.append(row)

    # ---------- B) OFF-DIAGONAL: compare different configs (window vs window) ----------
    # Compare each pair of configs once (cx < cy) to avoid duplicates
    configs_sorted = sorted(configs)
    for i in range(len(configs_sorted)):
        cx = configs_sorted[i]
        wx_list = sorted(windows_by_config.get(cx, set()))
        if not wx_list:
            continue

        for j in range(i + 1, len(configs_sorted)):
            cy = configs_sorted[j]
            wy_list = sorted(windows_by_config.get(cy, set()))
            if not wy_list:
                continue

            for wx in wx_list:
                runs_x = runs_by.get((cx, wx), [])
                if not runs_x:
                    continue

                for wy in wy_list:
                    runs_y = runs_by.get((cy, wy), [])
                    if not runs_y:
                        continue

                    # ---- IMPORTANT CHANGE ----
                    # Use run x run but cap to avoid explosion
                    # You can increase cap if needed
                    cap = 3  # compare only first 3 runs of each side (fast + fills heatmap)
                    runs_x_cap = runs_x[:cap]
                    runs_y_cap = runs_y[:cap]

                    for rx in runs_x_cap:
                        sx = idx.get((cx, wx, rx))
                        if sx is None:
                            continue

                        for ry in runs_y_cap:
                            sy = idx.get((cy, wy, ry))
                            if sy is None:
                                continue

                            common = sx.index.intersection(sy.index)
                            if len(common) < min_common_terms:
                                continue

                            lx = sx.loc[common].to_numpy()
                            ly = sy.loc[common].to_numpy()

                            try:
                                metrics = _compare_metrics(lx, ly)
                            except ValueError:
                                continue

                            row = {
                                "config_x": cx,
                                "config_y": cy,
                                "run_x": rx,
                                "run_y": ry,
                                "window_x": wx,
                                "window_y": wy,
                                "model_x": model_name,
                                "model_y": model_name,
                                "n_common_terms": int(len(common)),
                            }
                            row.update(metrics)
                            rows.append(row)

                            # Add symmetric entry to make heatmap full
                            row2 = dict(row)
                            row2["config_x"], row2["config_y"] = (
                                row["config_y"],
                                row["config_x"],
                            )
                            row2["window_x"], row2["window_y"] = (
                                row["window_y"],
                                row["window_x"],
                            )
                            row2["run_x"], row2["run_y"] = (
                                row["run_y"],
                                row["run_x"],
                            )
                            rows.append(row2)

    return rows


def compute_global_analysis(
    seed: int,
    n_samples: int,
    base_conf_dir: Path = Path("../outputs/conf"),
    base_analyses_dir: Path = Path("../outputs/analyses"),
):
    """
    Global comparisons among ALL configs with same seed + n_samples.
    Runs: sbm_vs_sbm, sbm_vs_w2v, w2v_vs_w2v.

    Optimized behavior:
      - only same-window comparisons (window_x == window_y)
      - intra-modal compares run pairs within same window (run_x < run_y)
      - cross-modal compares run_x (A) vs run_y (B) within same window
    """

    print(f"\n[ANALYSIS] Procurando TODAS as configs:")
    print(f"  seed={seed}, samples={n_samples}")

    base_conf_dir = Path(base_conf_dir)
    base_analyses_dir = Path(base_analyses_dir)
    base_analyses_dir.mkdir(parents=True, exist_ok=True)

    configs_found = find_all_configs_by_corpus(base_conf_dir, seed, n_samples)

    if not configs_found:
        print("[ERROR] Nenhuma config encontrada", file=sys.stderr)
        return False

    print(f"\n[FOUND] {len(configs_found)} configs encontradas:")
    for idx in sorted(configs_found.keys()):
        cfg = configs_found[idx]
        print(
            f"  Config {idx:04d}: {cfg['model']} | windows: {cfg['windows']}"
        )

    sbm_configs = {
        idx: cfg for idx, cfg in configs_found.items() if cfg["model"] == "sbm"
    }
    w2v_configs = {
        idx: cfg
        for idx, cfg in configs_found.items()
        if cfg["model"] == "w2v+kmeans"
    }

    print(f"\n  SBM configs: {list(sbm_configs.keys())}")
    print(f"  W2V configs: {list(w2v_configs.keys())}")

    # Load data for all configs
    all_data = {}
    for config_idx, cfg in configs_found.items():
        data = load_partition_data(cfg["config_dir"])
        if data is not None and not data.empty:
            all_data[config_idx] = data
        else:
            print(f"    [WARN] Nenhum parquet em config {config_idx:04d}")

    if not all_data:
        print("[ERROR] Sem dados para análise", file=sys.stderr)
        return False

    comparisons = [
        ("sbm", "sbm", "sbm_vs_sbm"),
        ("sbm", "w2v+kmeans", "sbm_vs_w2v"),
        ("w2v+kmeans", "w2v+kmeans", "w2v_vs_w2v"),
    ]

    all_success = True

    for model_x, model_y, comparison_name in comparisons:
        print(f"\n[COMPARE] {comparison_name}")

        valid_configs_x = (
            list(sbm_configs.keys())
            if model_x == "sbm"
            else list(w2v_configs.keys())
        )
        valid_configs_y = (
            list(sbm_configs.keys())
            if model_y == "sbm"
            else list(w2v_configs.keys())
        )

        print(f"    Valid configs X ({model_x}): {valid_configs_x}")
        print(f"    Valid configs Y ({model_y}): {valid_configs_y}")

        # Build dataframes per side
        dfs_x = [all_data[c].copy() for c in valid_configs_x if c in all_data]
        dfs_y = [all_data[c].copy() for c in valid_configs_y if c in all_data]

        if not dfs_x or not dfs_y:
            print(f"  [SKIP] Sem dados para {model_x} ou {model_y}")
            all_success = False
            continue

        data_x = pd.concat(dfs_x, ignore_index=True)
        data_y = pd.concat(dfs_y, ignore_index=True)

        # Normalize window type
        data_x["window"] = data_x["window"].astype(str)
        data_y["window"] = data_y["window"].astype(str)

        configs_x = sorted(data_x["config"].unique().tolist())
        configs_y = sorted(data_y["config"].unique().tolist())

        print(f"    Configs X ({model_x}): {configs_x}")
        print(f"    Configs Y ({model_y}): {configs_y}")

        # Build fast indices
        idx_x, runs_by_x = build_term_index(data_x)
        idx_y, runs_by_y = build_term_index(data_y)

        # Compute comparisons (optimized)
        if model_x == model_y:
            rows = compare_intra_windows_full_heatmap(
                idx=idx_x,
                runs_by=runs_by_x,
                configs=configs_x,
                model_name=model_x,
                min_common_terms=2,
            )
        else:
            rows = compare_cross_all_windows_aligned_runs(
                idx_a=idx_x,
                runs_by_a=runs_by_x,
                configs_a=configs_x,
                model_a=model_x,
                idx_b=idx_y,
                runs_by_b=runs_by_y,
                configs_b=configs_y,
                model_b=model_y,
                min_common_terms=2,
            )

        if not rows:
            print(f"  [WARN] Sem comparações válidas para {comparison_name}")
            all_success = False
            continue

        df_result = pd.DataFrame(rows)

        # Use all relevant configs
        all_config_indices = sorted(set(configs_x + configs_y))

        ana_mgr = analysis_manager.AnalysisManager(base_analyses_dir)
        analysis_dir, analysis_idx = ana_mgr.find_or_create_analysis_dir(
            all_config_indices, comparison_name
        )

        results_parquet = analysis_dir / "results.parquet"
        df_result.to_parquet(results_parquet, engine="pyarrow")
        print(f"  [SAVED] results.parquet: {results_parquet}")

        ana_mgr.save_analysis_config(
            analysis_dir,
            all_config_indices,
            comparison_name,
            corpus_seed=seed,
            n_samples=n_samples,
            graph_type=None,
        )

        df_summary = df_result.drop(
            columns=["model_x", "model_y"], errors="ignore"
        )
        ana_mgr.save_analysis_results(
            analysis_dir,
            df_summary,
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
        print("\n✓ Análises concluídas com sucesso!")
        sys.exit(0)
    else:
        print("\n✗ Alguns erros ocorreram")
        sys.exit(1)
