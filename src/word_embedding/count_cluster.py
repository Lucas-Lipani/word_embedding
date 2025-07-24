from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Caminho ajustado para execução a partir de src/
base_path = Path("../outputs/partitions")

for sample_dir in base_path.glob("*"):
    if not sample_dir.is_dir() or not sample_dir.name.isdigit():
        continue
    print(f"Amostra: {sample_dir.name}")
    for seed_dir in sample_dir.glob("seed_*"):
        print(f"  Seed: {seed_dir.name}")
        rows = []

        for model_dir in seed_dir.glob("*_J*"):
            match = re.match(r"(sbm|w2v)_J(.+)", model_dir.name)
            if not match:
                continue
            model, window = match.groups()
            for pf in model_dir.glob("partitions_run*.parquet"):
                run = pf.stem.split("run")[1]
                df = pd.read_parquet(pf)
                partitions = df["label"].nunique()
                rows.append(
                    {
                        "model": model,
                        "window": window,
                        "run": run,
                        "partitions": partitions,
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            df.sort_values(["model", "window", "run"], inplace=True)

            analysis_dir = seed_dir / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)

            output_file = analysis_dir / "cluster_counts.csv"
            df.to_csv(output_file, index=False)
            print(f"  Arquivo salvo em: {output_file}")

            # === Gera gráfico de dispersão ===
            plt.figure(figsize=(10, 6))
            ax = sns.scatterplot(
                data=df,
                x="window",
                y="partitions",
                hue="model",
                style="model",
                alpha=0.7,
                s=60,
            )

            # Média por modelo e janela
            mean_df = df.groupby(["model", "window"])["partitions"].mean().reset_index()
            for _, row in mean_df.iterrows():
                ax.text(
                    row["window"],
                    row["partitions"] + 0.3,
                    f'{row["partitions"]:.1f}',
                    ha="center",
                    fontsize=9,
                    color="black",
                )

            # Média geral por modelo
            overall_means = df.groupby("model")["partitions"].mean().reset_index()
            colors = dict(
                zip(
                    df["model"].unique(),
                    sns.color_palette(n_colors=len(df["model"].unique())),
                )
            )

            for _, row in overall_means.iterrows():
                ax.axhline(
                    y=row["partitions"],
                    color=colors[row["model"]],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.5,
                )
                x_middle = df["window"].unique()[len(df["window"].unique()) // 2]

                ax.text(
                    x=x_middle,
                    y=row["partitions"] + 8,
                    s=f"Média geral: {round(row['partitions'])}",
                    color="black",
                    fontsize=10,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            plt.title(f"Número de partições — {sample_dir.name} / {seed_dir.name}")
            plt.ylabel("Número de partições")
            plt.xlabel("Janela")
            plt.legend(title="Modelo")
            plt.tight_layout()

            out_png = analysis_dir / "cluster_counts_plot.png"
            plt.savefig(out_png)
            plt.close()
            print(f"  Gráfico salvo em: {out_png}")

        else:
            print(f"  Nenhum dado encontrado para {seed_dir}")
