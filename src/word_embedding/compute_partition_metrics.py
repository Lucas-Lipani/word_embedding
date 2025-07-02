from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score
from graph_tool.all import variation_information, mutual_information, partition_overlap


def compare_labels(a, b):
    vi = variation_information(a, b)
    mi = mutual_information(a, b)
    po = partition_overlap(a, b)
    nmi = po[2] if isinstance(po, (tuple, list, np.ndarray)) else po
    ari = adjusted_rand_score(a, b)
    return vi, nmi, ari


def load_all_runs(seed_path):
    """
    Reconstrói os partitions_runXXX.parquet agrupando por janela e modelo
    Retorna: dict(run_id: DataFrame completo com todas janelas e modelos)
    """
    run_data = defaultdict(list)
    for model_dir in seed_path.iterdir():
        if not model_dir.is_dir():
            continue
        model, janela = model_dir.name.split("_J")
        for part_file in model_dir.glob("partitions_run*.parquet"):
            run_id = part_file.stem.split("run")[1]
            df = pd.read_parquet(part_file)
            df["window"] = janela
            df["model"] = model
            run_data[run_id].append(df)

    # Agrupar os dfs por execução
    all_runs = {}
    for run_id, dfs in run_data.items():
        full_df = pd.concat(dfs, ignore_index=True)
        all_runs[run_id] = full_df
    return all_runs


def load_partition_by_model(df, model):
    model_df = df[df["model"] == model]
    grouped = model_df.groupby("window")
    return {
        str(w): g.set_index("term")["label"].to_dict()
        for w, g in grouped
    }


def compare_all_pairs(dict_a, dict_b, model_a, model_b):
    rows = []
    for win_i, labels_i in dict_a.items():
        for win_j, labels_j in dict_b.items():
            common = set(labels_i) & set(labels_j)
            if len(common) < 2:
                vi = nmi = ari = np.nan
            else:
                la = [labels_i[t] for t in common]
                lb = [labels_j[t] for t in common]
                vi, nmi, ari = compare_labels(la, lb)

            rows.append({
                "sbm_window": win_i,
                "w2v_window": win_j,
                "vi": vi, "nmi": nmi, "ari": ari,
                "model_pair": f"{model_a}_{model_b}"
            })
    return rows


def process_seed_folder(seed_folder):
    seed_folder = Path(seed_folder)
    print(f"\n📁 Processando: {seed_folder}")

    runs_data = load_all_runs(seed_folder)
    all_metrics = []

    for run_id, df in runs_data.items():
        sbm = load_partition_by_model(df, "sbm")
        w2v = load_partition_by_model(df, "w2v")

        rows = []
        rows += compare_all_pairs(sbm, w2v, "sbm", "w2v")
        rows += compare_all_pairs(sbm, sbm, "sbm", "sbm")
        rows += compare_all_pairs(w2v, w2v, "w2v", "w2v")

        df_run = pd.DataFrame(rows)
        out_file = seed_folder / f"metrics_run{run_id}.parquet"
        df_run.to_parquet(out_file, engine="pyarrow")
        print(f"[✔] Run {run_id} → {out_file.name}")
        all_metrics.append(df_run)

    if not all_metrics:
        print("[⚠] Nenhuma métrica gerada.")
        return

    df_all = pd.concat(all_metrics, ignore_index=True)
    df_all["sbm_window"] = df_all["sbm_window"].astype(str)
    df_all["w2v_window"] = df_all["w2v_window"].astype(str)

    mean_df = (
        df_all
        .groupby(["sbm_window", "w2v_window", "model_pair"])[["vi", "nmi", "ari"]]
        .mean()
        .reset_index()
    )

    mean_file = seed_folder / "running_mean.parquet"
    mean_df.to_parquet(mean_file, engine="pyarrow")
    print(f"[✔] Média final → {mean_file.name}")


if __name__ == "__main__":
    base_dir = Path("../outputs/partitions").resolve()
    seed_folders = sorted(base_dir.glob("*/*/"))

    for folder in seed_folders:
        if any(folder.glob("*/partitions_run*.parquet")):
            process_seed_folder(folder)
