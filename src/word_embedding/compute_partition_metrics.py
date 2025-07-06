# compute_partition_metrics.py — generate running_mean.parquet files only
# Scans raw partitions_run*.parquet produced by experiments and computes
# mean VI / NMI / ARI matrices for each seed and each comparison.
# Output is stored in:
#   outputs/partitions/<n_samples>/<seed_xxxx>/analysis/<model_x>_vs_<model_y>/running_mean.parquet

from pathlib import Path
import pandas as pd
import numpy as np
from graph_tool.all import variation_information, partition_overlap
from sklearn.metrics import adjusted_rand_score

# ----------------------------------------------------------------------------

def _window_sort_key(w):
    return float('inf') if w == 'full' else int(w)


def _compare_metrics(labels_a: np.ndarray, labels_b: np.ndarray):
    vi = variation_information(labels_a, labels_b)
    po = partition_overlap(labels_a, labels_b)
    nmi = po[2] if isinstance(po, (tuple, list, np.ndarray)) else float(po)
    ari = adjusted_rand_score(labels_a, labels_b)
    return vi, nmi, ari


def compute_seed(seed_dir: Path, model_x: str, model_y: str):
    out_root = seed_dir / "analysis" / f"{model_x}_vs_{model_y}"
    out_root.mkdir(parents=True, exist_ok=True)

    part_files = list(seed_dir.rglob("partitions_run*.parquet"))
    if not part_files:
        print(f"  [WARN] Nenhum partitions_run* encontrado em {seed_dir}")
        return

    dfs = []
    for pf in part_files:
        try:
            run_idx = int(pf.stem.split("_run")[1].split("_")[0])
        except ValueError:
            continue
        df = pd.read_parquet(pf)
        df["run"] = run_idx
        dfs.append(df)

    if not dfs:
        print(f"  [WARN] Partitions mal nomeados em {seed_dir}")
        return

    data = pd.concat(dfs, ignore_index=True)
    data["window"] = data["window"].astype(str)

    runs = sorted(data["run"].unique())
    windows = sorted(data["window"].unique(), key=_window_sort_key)

    row_key = f"{model_x}_row_window" if model_x == model_y else f"{model_x}_window"
    col_key = f"{model_y}_col_window" if model_x == model_y else f"{model_y}_window"

    rows = []
    for wx in windows:
        for wy in windows:
            for r1 in runs:
                df_x = data[(data["model"] == model_x) & (data["window"] == wx) & (data["run"] == r1)]
                if df_x.empty:
                    continue
                for r2 in runs:
                    if model_x == model_y and r1 == r2:
                        continue
                    df_y = data[(data["model"] == model_y) & (data["window"] == wy) & (data["run"] == r2)]
                    if df_y.empty:
                        continue
                    common = set(df_x["term"]).intersection(df_y["term"])
                    if len(common) < 2:
                        continue
                    vi, nmi, ari = _compare_metrics(
                        df_x.set_index("term").loc[list(common)]["label"].values,
                        df_y.set_index("term").loc[list(common)]["label"].values,
                    )
                    rows.append({row_key: wx, col_key: wy, "vi": vi, "nmi": nmi, "ari": ari})

    if not rows:
        print(f"  [WARN] Nenhum termo comum para {model_x} vs {model_y} em {seed_dir.name}")
        return

    mean_df = (
        pd.DataFrame(rows)
        .groupby([row_key, col_key])[['vi', 'nmi', 'ari']].mean()
        .reset_index()
    )

    mean_df.to_parquet(out_root / "running_mean.parquet", engine="pyarrow")
    print(f"    running_mean salvo: {out_root / 'running_mean.parquet'}")

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    base = Path("../outputs/partitions")
    comparisons = [("sbm", "w2v"), ("sbm", "sbm"), ("w2v", "w2v")]

    for sample_dir in sorted(base.glob("*")):
        if not sample_dir.is_dir():
            continue
        print(f"Amostras: {sample_dir.name}")

        for seed_dir in sorted(sample_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            print(f"  Seed: {seed_dir.name}")
            for model_x, model_y in comparisons:
                print(f"    {model_x.upper()} × {model_y.upper()}")
                compute_seed(seed_dir, model_x, model_y)
