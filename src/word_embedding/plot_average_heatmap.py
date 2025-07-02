from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc


def plot_heatmap(df, tipo="sbm_w2v", metric="nmi", folder=Path(".")):
    df_filtered = df[df["model_pair"] == tipo]
    if metric not in df_filtered.columns:
        print(f"[⚠] Métrica '{metric}' não encontrada no dataframe.")
        return

    df_pivot = df_filtered.pivot(index="sbm_window", columns="w2v_window", values=metric)
    df_pivot = df_pivot.sort_index().sort_index(axis=1)

    if df_pivot.empty:
        print(f"[⚠] Nenhum dado válido para {metric.upper()} ({tipo}) em {folder}")
        return

    plt.figure(figsize=(len(df_pivot.columns) + 1, len(df_pivot.index) + 1))
    sns.heatmap(
        df_pivot,
        annot=True,
        fmt=".2f",
        cmap=getattr(cc.cm, "bgy" if metric != "vi" else "fire"),
        linewidths=0.5,
        cbar=True,
    )
    plt.title(f"{metric.upper()} – {tipo.replace('_', ' × ').upper()}")
    plt.xlabel("Word2Vec Window")
    plt.ylabel("SBM Window")
    plt.tight_layout()

    out_file = Path(folder) / f"mean_{metric}_{tipo}.png"
    plt.savefig(out_file)
    print(f"[✔] Heatmap salvo: {out_file}")
    plt.close()


def main():
    base_path = Path("../outputs/partitions").resolve()
    seed_folders = sorted(base_path.glob("*/*/"))

    for seed_path in seed_folders:
        mean_file = seed_path / "running_mean.parquet"
        if not mean_file.exists():
            continue

        print(f"\n📈 Gerando gráficos para: {seed_path}")
        df = pd.read_parquet(mean_file)

        for tipo in ["sbm_w2v", "sbm_sbm", "w2v_w2v"]:
            if tipo not in df["model_pair"].unique():
                continue
            for metric in ["nmi", "vi", "ari"]:
                plot_heatmap(df, tipo=tipo, metric=metric, folder=seed_path)


if __name__ == "__main__":
    main()
