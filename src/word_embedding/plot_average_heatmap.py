# plot_average_robust.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

def regenerate_mean_if_needed(folder):
    dir_path = Path(folder)
    mean_file = dir_path / "running_mean.parquet"
    if mean_file.exists():
        try:
            df = pd.read_parquet(mean_file)
            if df[["vi", "nmi", "ari"]].dropna().empty:
                raise ValueError("Arquivo existente, mas vazio ou inválido.")
            return True
        except Exception as e:
            print(f"[⚠] Problema ao ler média existente: {e}")
    
    print(f"🔄 Regenerando média para: {folder}")
    files = sorted(dir_path.glob("metrics_run*.parquet"))
    if not files:
        print(f"[⚠] Nenhum metrics_run*.parquet encontrado em {folder}")
        return False
    try:
        all_dfs = [pd.read_parquet(p).set_index(["sbm_window", "w2v_window"]) for p in files]
        mean_df = sum(all_dfs) / len(all_dfs)
        mean_df.reset_index().to_parquet(mean_file, engine="pyarrow")
        print(f"[✔] Média salva: {mean_file}")
        return True
    except Exception as e:
        print(f"[❌] Erro ao calcular média: {e}")
        return False

def plot_mean_heatmap(folder, metric="nmi", cmap="bgy", save=True):
    file = Path(folder) / "running_mean.parquet"
    if not file.exists():
        print(f"[⚠] Arquivo não encontrado: {file}")
        return

    df = pd.read_parquet(file)
    if metric not in df.columns:
        print(f"[⚠] Métrica '{metric}' não encontrada no arquivo {file.name}")
        return

    df_filtered = df[["sbm_window", "w2v_window", metric]].dropna()
    df_filtered["sbm_window"] = df_filtered["sbm_window"].astype(str)
    df_filtered["w2v_window"] = df_filtered["w2v_window"].astype(str)

    pivot = df_filtered.pivot(index="sbm_window", columns="w2v_window", values=metric)
    pivot = pivot.sort_index().sort_index(axis=1)

    if pivot.empty:
        print(f"[⚠] Nenhum dado válido para {metric.upper()} em {folder}")
        return

    plt.figure(figsize=(len(pivot.columns) + 1, len(pivot.index) + 1))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=getattr(cc.cm, cmap),
        linewidths=0.5,
        cbar=True
    )
    plt.title(f"Mean {metric.upper()} across runs\n({Path(folder).name})")
    plt.xlabel("Word2Vec Window")
    plt.ylabel("SBM Window")
    plt.tight_layout()

    if save:
        out_file = Path(folder) / f"mean_{metric}.png"
        plt.savefig(out_file)
        print(f"[✔] Heatmap salvo: {out_file}")
    else:
        plt.show()

if __name__ == "__main__":
    base_path = Path("../outputs/window")
    for folder in sorted(base_path.glob("*s_*w")):
        if regenerate_mean_if_needed(folder):
            for metric in ["nmi", "vi", "ari"]:
                plot_mean_heatmap(folder, metric=metric, cmap="bgy", save=True)
