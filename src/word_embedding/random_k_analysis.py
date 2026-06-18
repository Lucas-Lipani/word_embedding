from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis_manager, compute_analysis


METRIC_COLUMNS = [
    "vi", "nvi", "po", "npo", "mi", "ami",
    "nmi", "anmi", "ari", "rmi", "nrmi",
]


def _window_sort_key(value):
    text = str(value)
    if text.lower() == "full":
        return (1, text)
    try:
        return (0, int(text))
    except (TypeError, ValueError):
        return (0, text)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _same_graph_signature(
    cfg: dict,
    *,
    seed: int,
    samples: int,
    graph_type: str,
    nested: bool,
    layered: bool,
    edge_weighting: str,
    window_size: str,
    context: bool,
    model_kind: str,
    assignment_mode: str | None = None,
    repeats: int | None = None,
) -> bool:
    corpus = cfg.get("corpus", {})
    graph = cfg.get("graph", {})
    model = cfg.get("model", {})

    if corpus.get("seed") != seed:
        return False
    if corpus.get("number_of_documents") != samples:
        return False
    if graph.get("graph_type") != graph_type:
        return False
    if (graph.get("sbm_variant", "flat") == "nested") != nested:
        return False
    if graph.get("sbm_layered", False) != layered:
        return False
    if graph.get("edge_weighting", "inverse_window_size") != edge_weighting:
        return False
    if str(graph.get("window_size")) != str(window_size):
        return False
    if graph.get("context", False) != context:
        return False
    if model.get("kind") != model_kind:
        return False

    if assignment_mode is not None:
        if model.get("assignment_mode") != assignment_mode:
            return False
    if repeats is not None:
        if model.get("repeats") != repeats:
            return False

    return True


def _find_or_create_random_config_dir(
    base_conf_dir: Path,
    *,
    seed: int,
    samples: int,
    graph_type: str,
    nested: bool,
    layered: bool,
    edge_weighting: str,
    window_size: str,
    context: bool,
    assignment_mode: str,
    repeats: int,
    random_seed: int,
    source_sbm_config: int,
) -> Path:
    for config_file in sorted(base_conf_dir.glob("????/config.json")):
        try:
            cfg = _load_json(config_file)
        except Exception:
            continue

        if _same_graph_signature(
            cfg,
            seed=seed,
            samples=samples,
            graph_type=graph_type,
            nested=nested,
            layered=layered,
            edge_weighting=edge_weighting,
            window_size=window_size,
            context=context,
            model_kind="random_k",
            assignment_mode=assignment_mode,
            repeats=repeats,
        ):
            print(f"[CONFIG] Reusing random_k config: {config_file.parent.name}")
            return config_file.parent

    existing = sorted(base_conf_dir.glob("????"))
    next_idx = max([int(path.name) for path in existing], default=0) + 1
    config_dir = base_conf_dir / f"{next_idx:04d}"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "timestamp": datetime.now().isoformat(),
        "corpus": {
            "seed": seed,
            "number_of_documents": samples,
        },
        "graph": {
            "graph_type": graph_type,
            "sbm_variant": "nested" if nested else "flat",
            "sbm_layered": layered,
            "fixed_n_blocks": None,
            "window_size": window_size,
            "context": context,
            "edge_weighting": edge_weighting,
        },
        "model": {
            "kind": "random_k",
            "assignment_mode": assignment_mode,
            "repeats": repeats,
            "random_seed": random_seed,
            "source_model": "sbm",
            "source_sbm_config": source_sbm_config,
            "k_source": "number_of_distinct_SBM_term_labels_per_run",
        },
    }
    _save_json(config_dir / "config.json", config_data)
    print(f"[CONFIG] Created random_k config: {config_dir.name}")
    return config_dir


def _extract_terms_and_k_from_sbm_run(partition_file: Path) -> tuple[pd.DataFrame, int]:
    df = pd.read_parquet(partition_file)

    terms = df[(df["tipo"] == 1) & (df["term"].notna())].copy()
    if terms.empty:
        raise ValueError(f"No term rows found in {partition_file}")

    terms = terms[["window", "term", "vertex", "freq", "label"]].copy()
    terms["window"] = terms["window"].astype(str)
    terms["term"] = terms["term"].astype(str)
    terms["vertex"] = terms["vertex"].astype(str)
    if "freq" not in terms.columns:
        terms["freq"] = 1
    terms["freq"] = terms["freq"].fillna(1).astype(int)

    # One row per term. If duplicated, keep the first occurrence to avoid double labels.
    terms = terms.drop_duplicates(subset=["term"], keep="first").reset_index(drop=True)
    k = int(terms["label"].nunique())
    if k < 1:
        raise ValueError(f"Invalid k={k} in {partition_file}")

    return terms, k


def _assign_random_labels(
    terms: pd.Series,
    k: int,
    rng: np.random.Generator,
    mode: str,
) -> np.ndarray:
    n_terms = len(terms)
    if n_terms == 0:
        return np.array([], dtype=int)

    k = min(int(k), n_terms)

    if mode == "balanced":
        labels = np.arange(n_terms, dtype=int) % k
        rng.shuffle(labels)
        return labels

    if mode == "pure":
        # Pure random assignment, but still guarantees that the requested k
        # groups are represented at least once when n_terms >= k.
        # Without this safeguard, an iid draw may accidentally leave empty groups,
        # which would no longer be a partition into exactly k groups.
        base_labels = np.arange(k, dtype=int)
        if n_terms > k:
            extra_labels = rng.integers(0, k, size=n_terms - k, dtype=int)
            labels = np.concatenate([base_labels, extra_labels])
        else:
            labels = base_labels
        rng.shuffle(labels)
        return labels

    raise ValueError(f"Unsupported assignment mode: {mode}")


def _save_random_run(
    random_config_dir: Path,
    random_run_idx: int,
    random_df: pd.DataFrame,
    *,
    source_run: int,
    null_repeat: int,
    k: int,
    assignment_mode: str,
) -> None:
    run_dir = random_config_dir / "run" / f"{random_run_idx:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    random_df.to_parquet(run_dir / "partition.parquet", engine="pyarrow")

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "model": "random_k",
        "random_k": {
            "number_of_clusters": int(k),
            "assignment_mode": assignment_mode,
            "source_run": int(source_run),
            "null_repeat": int(null_repeat),
            "n_terms": int(len(random_df)),
        },
    }
    _save_json(run_dir / "results.json", results_data)


def generate_random_k_partitions(
    *,
    seed: int,
    samples: int,
    graph_type: str | None,
    layered: bool,
    nested: bool,
    edge_weighting: str,
    context: bool | None,
    windows_filter: set[str] | None,
    repeats: int,
    assignment_mode: str,
    random_seed: int,
    base_conf_dir: Path,
    overwrite: bool,
) -> list[int]:
    configs_found = compute_analysis.find_all_configs_by_corpus(
        base_conf_dir=base_conf_dir,
        seed=seed,
        n_samples=samples,
        graph_type=graph_type,
        layered=layered,
        nested=nested,
        edge_weighting=edge_weighting,
    )

    sbm_configs = {
        idx: cfg for idx, cfg in configs_found.items() if cfg["model"] == "sbm"
    }
    if not sbm_configs:
        raise RuntimeError("No SBM configs found for the requested signature.")

    random_config_indices: list[int] = []

    for sbm_config_idx, sbm_cfg in sorted(sbm_configs.items()):
        sbm_config_dir = sbm_cfg["config_dir"]
        sbm_config = _load_json(sbm_config_dir / "config.json")
        graph = sbm_config.get("graph", {})

        cfg_window = str(graph.get("window_size"))
        cfg_context = graph.get("context", False)
        if context is not None and cfg_context != context:
            continue
        if windows_filter is not None and cfg_window not in windows_filter:
            continue

        random_config_dir = _find_or_create_random_config_dir(
            base_conf_dir,
            seed=seed,
            samples=samples,
            graph_type=graph.get("graph_type"),
            nested=graph.get("sbm_variant", "flat") == "nested",
            layered=graph.get("sbm_layered", False),
            edge_weighting=graph.get("edge_weighting", edge_weighting),
            window_size=cfg_window,
            context=cfg_context,
            assignment_mode=assignment_mode,
            repeats=repeats,
            random_seed=random_seed,
            source_sbm_config=sbm_config_idx,
        )
        random_config_indices.append(int(random_config_dir.name))

        run_dirs = sorted((sbm_config_dir / "run").glob("????"))
        for run_dir in run_dirs:
            partition_file = run_dir / "partition.parquet"
            if not partition_file.exists():
                continue

            source_run = int(run_dir.name)
            terms_df, k = _extract_terms_and_k_from_sbm_run(partition_file)
            window_label = str(terms_df["window"].iloc[0])

            for repeat_idx in range(1, repeats + 1):
                random_run_idx = (source_run - 1) * repeats + repeat_idx
                out_file = random_config_dir / "run" / f"{random_run_idx:04d}" / "partition.parquet"
                if out_file.exists() and not overwrite:
                    continue

                # Make the null partition reproducible but different for each config/run/repeat.
                local_seed = random_seed + (sbm_config_idx * 1_000_000) + (source_run * 10_000) + repeat_idx
                rng = np.random.default_rng(local_seed)
                labels = _assign_random_labels(terms_df["term"], k, rng, assignment_mode)

                random_df = pd.DataFrame(
                    {
                        "window": window_label,
                        "model": "random_k",
                        "vertex": terms_df["vertex"].to_numpy(),
                        "tipo": 1,
                        "label": labels.astype(int),
                        "doc_id": None,
                        "term": terms_df["term"].to_numpy(),
                        "freq": terms_df["freq"].to_numpy(),
                        "source_config": sbm_config_idx,
                        "source_run": source_run,
                        "null_repeat": repeat_idx,
                        "k_source": k,
                    }
                )

                _save_random_run(
                    random_config_dir,
                    random_run_idx,
                    random_df,
                    source_run=source_run,
                    null_repeat=repeat_idx,
                    k=k,
                    assignment_mode=assignment_mode,
                )

        print(f"[OK] random_k generated for SBM config {sbm_config_idx:04d} -> random config {int(random_config_dir.name):04d}")

    return sorted(set(random_config_indices))


def _load_model_data_by_kind(
    configs_found: dict,
    model_kind: str,
) -> tuple[pd.DataFrame | None, list[int]]:
    selected = {
        idx: cfg for idx, cfg in configs_found.items() if cfg["model"] == model_kind
    }
    dfs = []
    for idx, cfg in selected.items():
        data = compute_analysis.load_partition_data(cfg["config_dir"])
        if data is not None and not data.empty:
            dfs.append(data)
    if not dfs:
        return None, []
    return pd.concat(dfs, ignore_index=True), sorted(selected.keys())


def _load_random_data(
    base_conf_dir: Path,
    *,
    seed: int,
    samples: int,
    graph_type: str | None,
    layered: bool,
    nested: bool,
    edge_weighting: str,
    context: bool | None,
    assignment_mode: str,
    repeats: int,
    windows_filter: set[str] | None,
) -> tuple[pd.DataFrame | None, list[int]]:
    dfs = []
    config_indices = []

    for config_file in sorted(base_conf_dir.glob("????/config.json")):
        try:
            cfg = _load_json(config_file)
        except Exception:
            continue

        corpus = cfg.get("corpus", {})
        graph = cfg.get("graph", {})
        model = cfg.get("model", {})

        if model.get("kind") != "random_k":
            continue
        if model.get("assignment_mode") != assignment_mode:
            continue
        if model.get("repeats") != repeats:
            continue
        if corpus.get("seed") != seed or corpus.get("number_of_documents") != samples:
            continue
        if graph_type is not None and graph.get("graph_type") != graph_type:
            continue
        if graph.get("sbm_layered", False) != layered:
            continue
        if (graph.get("sbm_variant", "flat") == "nested") != nested:
            continue
        if graph.get("edge_weighting", "inverse_window_size") != edge_weighting:
            continue
        if context is not None and graph.get("context", False) != context:
            continue
        if windows_filter is not None and str(graph.get("window_size")) not in windows_filter:
            continue

        config_dir = config_file.parent
        data = compute_analysis.load_partition_data(config_dir)
        if data is not None and not data.empty:
            dfs.append(data)
            config_indices.append(int(config_dir.name))

    if not dfs:
        return None, []

    return pd.concat(dfs, ignore_index=True), sorted(config_indices)


def _compare_model_to_random(
    model_df: pd.DataFrame,
    random_df: pd.DataFrame,
    *,
    model_name: str,
    comparison_name: str,
    same_window_only: bool = False,
) -> pd.DataFrame:
    rows = []

    model_terms = model_df[(model_df["tipo"] == 1) & (model_df["term"].notna())].copy()
    random_terms = random_df[(random_df["tipo"] == 1) & (random_df["term"].notna())].copy()

    model_terms["window"] = model_terms["window"].astype(str)
    random_terms["window"] = random_terms["window"].astype(str)

    required_random = {"source_run", "null_repeat", "k_source"}
    missing = required_random - set(random_terms.columns)
    if missing:
        raise ValueError(f"Random dataframe is missing columns: {missing}")

    model_groups = {
        (int(config), str(window), int(run)): sub.set_index("term")["label"]
        for (config, window, run), sub in model_terms.groupby(["config", "window", "run"], sort=False)
    }

    random_groups = {
        (
            int(config),
            str(window),
            int(source_run),
            int(null_repeat),
            int(run),
        ): sub.set_index("term")["label"]
        for (config, window, source_run, null_repeat, run), sub in random_terms.groupby(
            ["config", "window", "source_run", "null_repeat", "run"], sort=False
        )
    }

    for (model_config, window, model_run), model_series in model_groups.items():
        for (random_config, random_window, source_run, null_repeat, random_run), random_series in random_groups.items():
            if same_window_only and random_window != window:
                continue
            if source_run != model_run:
                continue

            common_terms = model_series.index.intersection(random_series.index)
            if len(common_terms) < 2:
                continue

            labels_model = model_series.loc[common_terms].to_numpy()
            labels_random = random_series.loc[common_terms].to_numpy()
            metrics = compute_analysis._compare_metrics(labels_model, labels_random)

            row = {
                "config_x": model_config,
                "config_y": random_config,
                "run_x": model_run,
                "run_y": random_run,
                "window_x": window,
                "window_y": random_window,
                "model_x": model_name,
                "model_y": "random_k",
                "comparison_type": comparison_name,
                "n_common_terms": int(len(common_terms)),
                "source_run": source_run,
                "null_repeat": null_repeat,
                "k_source": int(random_terms[(random_terms["config"] == random_config) & (random_terms["run"] == random_run)]["k_source"].iloc[0]),
            }
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)



def _compare_random_to_random(random_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare random_k partitions against each other.

    This uses the same intra-model logic as SBM×SBM and W2V×W2V:
    - diagonal cells compare different runs inside the same window/config;
    - off-diagonal cells compare runs between different window configs;
    - symmetric rows are added by compute_analysis.compare_intra_windows_full_heatmap.
    """
    random_terms = random_df[
        (random_df["tipo"] == 1) & (random_df["term"].notna())
    ].copy()

    if random_terms.empty:
        return pd.DataFrame()

    random_terms["window"] = random_terms["window"].astype(str)

    idx, runs_by = compute_analysis.build_term_index(random_terms)
    configs = sorted(random_terms["config"].astype(int).unique().tolist())

    rows = compute_analysis.compare_intra_windows_full_heatmap(
        idx=idx,
        runs_by=runs_by,
        configs=configs,
        model_name="random_k",
        min_common_terms=2,
    )

    if not rows:
        return pd.DataFrame()

    df_result = pd.DataFrame(rows)
    df_result["comparison_type"] = "random_k_vs_random_k"
    return df_result

def _save_analysis(
    df_result: pd.DataFrame,
    *,
    comparison_name: str,
    config_indices: list[int],
    seed: int,
    samples: int,
    base_analyses_dir: Path,
    graph_metadata: dict,
    windows: list[str],
) -> None:
    if df_result.empty:
        print(f"[WARN] No rows for {comparison_name}")
        return

    ana_mgr = analysis_manager.AnalysisManager(base_analyses_dir)
    analysis_dir, analysis_idx = ana_mgr.find_or_create_analysis_dir(
        config_indices,
        comparison_name,
    )

    df_result.to_parquet(analysis_dir / "results.parquet", engine="pyarrow")

    ana_mgr.save_analysis_config(
        analysis_dir,
        config_indices,
        comparison_name,
        corpus_seed=seed,
        n_samples=samples,
        graph_type=graph_metadata.get("graph_type"),
        sbm_variant=graph_metadata.get("sbm_variant"),
        sbm_layered=graph_metadata.get("sbm_layered", False),
        window_size=windows,
        context=graph_metadata.get("context", False),
        edge_weighting=graph_metadata.get("edge_weighting"),
    )

    summary_df = df_result.drop(
        columns=[
            "model_x", "model_y", "comparison_type",
            "source_run", "null_repeat", "k_source",
        ],
        errors="ignore",
    )
    ana_mgr.save_analysis_results(analysis_dir, summary_df, comparison_name)

    print(f"[ANALYSIS {analysis_idx:04d}] {comparison_name}: {len(df_result)} comparisons")


def compute_random_k_analyses(
    *,
    seed: int,
    samples: int,
    graph_type: str | None,
    layered: bool,
    nested: bool,
    edge_weighting: str,
    context: bool | None,
    windows_filter: set[str] | None,
    assignment_mode: str,
    repeats: int,
    base_conf_dir: Path,
    base_analyses_dir: Path,
    same_window_only: bool = False,
) -> None:
    configs_found = compute_analysis.find_all_configs_by_corpus(
        base_conf_dir=base_conf_dir,
        seed=seed,
        n_samples=samples,
        graph_type=graph_type,
        layered=layered,
        nested=nested,
        edge_weighting=edge_weighting,
    )

    if context is not None:
        configs_found = {
            idx: cfg for idx, cfg in configs_found.items()
            if _load_json(cfg["config_dir"] / "config.json").get("graph", {}).get("context", False) == context
        }

    if windows_filter is not None:
        configs_found = {
            idx: cfg for idx, cfg in configs_found.items()
            if any(str(w) in windows_filter for w in cfg.get("windows", []))
        }

    sbm_df, sbm_configs = _load_model_data_by_kind(configs_found, "sbm")
    w2v_df, w2v_configs = _load_model_data_by_kind(configs_found, "w2v+kmeans")
    random_df, random_configs = _load_random_data(
        base_conf_dir,
        seed=seed,
        samples=samples,
        graph_type=graph_type,
        layered=layered,
        nested=nested,
        edge_weighting=edge_weighting,
        context=context,
        assignment_mode=assignment_mode,
        repeats=repeats,
        windows_filter=windows_filter,
    )

    if random_df is None or random_df.empty:
        raise RuntimeError("No random_k partitions found. Run generation first.")

    graph_metadata = {}
    for cfg_idx in sbm_configs + random_configs + w2v_configs:
        cfg_dir = base_conf_dir / f"{cfg_idx:04d}"
        if (cfg_dir / "config.json").exists():
            cfg = _load_json(cfg_dir / "config.json")
            graph_metadata = cfg.get("graph", {})
            break

    windows = sorted(random_df["window"].astype(str).unique().tolist(), key=_window_sort_key)

    if sbm_df is not None and not sbm_df.empty:
        df_sbm_random = _compare_model_to_random(
            sbm_df,
            random_df,
            model_name="sbm",
            comparison_name="sbm_vs_random_k",
            same_window_only=same_window_only,
        )
        _save_analysis(
            df_sbm_random,
            comparison_name="sbm_vs_random_k",
            config_indices=sorted(set(sbm_configs + random_configs)),
            seed=seed,
            samples=samples,
            base_analyses_dir=base_analyses_dir,
            graph_metadata=graph_metadata,
            windows=windows,
        )

    if w2v_df is not None and not w2v_df.empty:
        df_w2v_random = _compare_model_to_random(
            w2v_df,
            random_df,
            model_name="w2v+kmeans",
            comparison_name="w2v_vs_random_k",
            same_window_only=same_window_only,
        )
        _save_analysis(
            df_w2v_random,
            comparison_name="w2v_vs_random_k",
            config_indices=sorted(set(w2v_configs + random_configs)),
            seed=seed,
            samples=samples,
            base_analyses_dir=base_analyses_dir,
            graph_metadata=graph_metadata,
            windows=windows,
        )

    df_random_random = _compare_random_to_random(random_df)
    _save_analysis(
        df_random_random,
        comparison_name="random_k_vs_random_k",
        config_indices=random_configs,
        seed=seed,
        samples=samples,
        base_analyses_dir=base_analyses_dir,
        graph_metadata=graph_metadata,
        windows=windows,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate one random_k baseline per existing SBM run by default, "
            "then compute SBM vs random_k, W2V vs random_k, and random_k vs random_k analyses."
        )
    )
    parser.add_argument("--seed", "-s", type=int, required=True, help="Corpus seed")
    parser.add_argument("--samples", type=int, required=True, help="Number of documents")
    parser.add_argument(
        "--graph-type",
        type=str,
        default=None,
        choices=[
            "Document-Window-Term",
            "Document-SlideWindow-Term",
            "Document-Context-Window-Term",
            "Document-Term",
        ],
        help="Graph type filter. If omitted, all graph types are considered.",
    )
    parser.add_argument(
        "--layered",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Filter layered SBM configs.",
    )
    parser.add_argument(
        "--nested",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Filter nested SBM configs.",
    )
    parser.add_argument(
        "--context",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional context filter.",
    )
    parser.add_argument(
        "--edge-weighting",
        choices=["uniform", "inverse_window_size"],
        default="inverse_window_size",
        help="Edge weighting filter.",
    )
    parser.add_argument(
        "--windows",
        nargs="*",
        default=None,
        help="Optional list of windows to process, for example: --windows 1 2 4 full",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Number of random_k partitions generated per original SBM run. "
            "Default 1 means one random baseline for each SBM run. Use values "
            "greater than 1 only to estimate an average random baseline."
        ),
    )
    parser.add_argument(
        "--assignment-mode",
        choices=["balanced", "pure"],
        default="balanced",
        help="balanced = near-equal random clusters; pure = iid random labels in [0, k).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Base random seed for reproducible null partitions.",
    )
    parser.add_argument(
        "--base-conf-dir",
        type=Path,
        default=Path("../outputs/conf"),
        help="Base config directory.",
    )
    parser.add_argument(
        "--base-analyses-dir",
        type=Path,
        default=Path("../outputs/analyses"),
        help="Base analyses directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing random_k partition files.",
    )
    parser.add_argument(
        "--same-window-only",
        action="store_true",
        help=(
            "Only compare a model window with the random_k generated from the same window. "
            "If omitted, the script also fills off-diagonal window comparisons."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    windows_filter = None
    if args.windows:
        windows_filter = {str(window) for window in args.windows}

    if args.repeats < 1:
        print("[ERROR] --repeats must be >= 1", file=sys.stderr)
        sys.exit(1)

    generated_configs = generate_random_k_partitions(
        seed=args.seed,
        samples=args.samples,
        graph_type=args.graph_type,
        layered=args.layered,
        nested=args.nested,
        edge_weighting=args.edge_weighting,
        context=args.context,
        windows_filter=windows_filter,
        repeats=args.repeats,
        assignment_mode=args.assignment_mode,
        random_seed=args.random_seed,
        base_conf_dir=args.base_conf_dir,
        overwrite=args.overwrite,
    )

    print(f"[OK] Random configs: {generated_configs}")

    compute_random_k_analyses(
        seed=args.seed,
        samples=args.samples,
        graph_type=args.graph_type,
        layered=args.layered,
        nested=args.nested,
        edge_weighting=args.edge_weighting,
        context=args.context,
        windows_filter=windows_filter,
        assignment_mode=args.assignment_mode,
        repeats=args.repeats,
        base_conf_dir=args.base_conf_dir,
        base_analyses_dir=args.base_analyses_dir,
        same_window_only=args.same_window_only,
    )


if __name__ == "__main__":
    main()
