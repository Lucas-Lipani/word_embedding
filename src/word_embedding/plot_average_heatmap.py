from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

def plot_mean_heatmap(folder, metric="nmi", cmap="bgy", save=False):
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
    base_path = Path("outputs/window")
    for folder in sorted(base_path.glob("*s_*w")):
        for metric in ["nmi", "vi", "ari"]:
            plot_mean_heatmap(folder, metric=metric, cmap="bgy", save=True)
