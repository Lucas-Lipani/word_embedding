# word_embedding/results_io.py
from pathlib import Path
import re
import pandas as pd

def _build_dir(base_out, n_samples, windows):
    win_tag = "-".join(map(str, windows))
    out_dir = Path(base_out) / f"{n_samples}s_{win_tag}w"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _next_run_idx(out_dir):
    # procura ficheiros do tipo *_runNNN_*.parquet
    expr = re.compile(r"_run(\d{3})_")
    idxs = []
    for p in out_dir.glob("*_run*.parquet"):
        m = expr.search(p.name)
        if m:
            idxs.append(int(m.group(1)))
    return (max(idxs) + 1) if idxs else 1

def save_run(out_base, n_samples, windows, metrics_df, partitions_df):
    """
    Salva Parquets usando modo 'x'. Se já existirem, falha (evita sobrescrever).
    Retorna o índice da execução salva.
    """
    out_dir = _build_dir(out_base, n_samples, windows)
    run_idx = _next_run_idx(out_dir)
    tag = f"_run{run_idx:03d}_"

    metrics_f = out_dir / f"metrics{tag}.parquet"
    partitions_f = out_dir / f"partitions{tag}.parquet"

    metrics_df.to_parquet(metrics_f, engine="pyarrow")
    partitions_df.to_parquet(partitions_f, engine="pyarrow")

    _update_running_mean(out_dir)

    return run_idx, metrics_f, partitions_f

def _update_running_mean(out_dir):
    """Concatena todos os metrics*.parquet, tira média e salva."""
    metrics_files = sorted(out_dir.glob("metrics_run*.parquet"))
    if not metrics_files:
        return
    all_metrics = [
        pd.read_parquet(p, engine="pyarrow").set_index(["sbm_window", "w2v_window"])
        for p in metrics_files
    ]
    mean_df = sum(all_metrics) / len(all_metrics)
    mean_df.reset_index().to_parquet(out_dir / "running_mean.parquet",
                                     engine="pyarrow")
