from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re


def parse_args():
    ap = argparse.ArgumentParser(
        description="Conta nº de partições por run/janela (por tipo) e salva CSV/plot."
    )
    ap.add_argument(
        "--seed", required=True, help="Seed alvo. Aceita '1234' ou 'seed_1234'."
    )
    ap.add_argument(
        "--samples",
        default=None,
        help="Pasta de amostras (ex.: 3). Se omitido, varre todas.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    base_path = Path("../outputs/partitions")

    # resolve seeds a visitar
    sample_dirs = (
        [base_path / args.samples]
        if args.samples
        else sorted([d for d in base_path.glob("*") if d.is_dir() and d.name.isdigit()])
    )

    seed_name = args.seed if str(args.seed).startswith("seed_") else f"seed_{args.seed}"

    for sample_dir in sample_dirs:
        if not sample_dir.is_dir():
            continue
        print(f"Amostra: {sample_dir.name}")

        seed_dir = sample_dir / seed_name
        if not seed_dir.is_dir():
            print(f"  [WARN] Seed {seed_dir.name} não encontrada em {sample_dir.name}")
            continue

        print(f"  Seed: {seed_dir.name}")
        rows = []

        # percorre modelos/janelas
        for model_dir in seed_dir.glob("*_J*"):
            m = re.match(r"(sbm|w2v)_J(.+)", model_dir.name)
            if not m:
                continue
            model, window = m.groups()

            for pf in sorted(model_dir.glob("partitions_run*.parquet")):
                # run: captura dígitos do final do nome
                mrun = re.search(r"run_?(\d+)", pf.stem)
                run = mrun.group(1) if mrun else pf.stem

                df = pd.read_parquet(pf)

                # Garante tipos consistentes
                if "label" in df.columns:
                    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(
                        "Int64"
                    )

                # Contagem de partições por TIPO
                if "tipo" in df.columns:
                    part_by_type = (
                        df.groupby("tipo")["label"].nunique(dropna=True)
                    ).reset_index()
                    for _, r in part_by_type.iterrows():
                        rows.append(
                            {
                                "model": model,
                                "window": str(window),
                                "run": run,
                                "tipo": int(r["tipo"]),
                                "partitions": int(r["label"]),
                            }
                        )
                else:
                    # fallback (antigo): sem coluna tipo, contar geral
                    rows.append(
                        {
                            "model": model,
                            "window": str(window),
                            "run": run,
                            "tipo": -1,
                            "partitions": df["label"].nunique(dropna=True),
                        }
                    )

        if not rows:
            print(f"  Nenhum dado encontrado para {seed_dir}")
            continue

        out_df = pd.DataFrame(rows).sort_values(["model", "window", "run", "tipo"])
        analysis_dir = seed_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        output_file = analysis_dir / "cluster_counts.csv"
        out_df.to_csv(output_file, index=False)
        print(f"  Arquivo salvo em: {output_file}")

        # === Gráfico (agregado por modelo/janela, independente do tipo) ===
        plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(
            data=out_df.groupby(["model", "window", "run"], as_index=False)[
                "partitions"
            ].sum(),
            x="window",
            y="partitions",
            hue="model",
            style="model",
            alpha=0.7,
            s=60,
        )

        # Médias por modelo/janela
        mean_df = out_df.groupby(["model", "window"])["partitions"].mean().reset_index()
        for _, row in mean_df.iterrows():
            ax.text(
                row["window"],
                row["partitions"] + 0.3,
                f'{row["partitions"]:.1f}',
                ha="center",
                fontsize=9,
                color="black",
            )

        plt.title(
            f"Nº de partições — {sample_dir.name} / {seed_dir.name} (soma sobre tipos)"
        )
        plt.ylabel("Nº de partições")
        plt.xlabel("Janela")
        plt.legend(title="Modelo")
        plt.tight_layout()

        out_png = analysis_dir / "cluster_counts_plot.png"
        plt.savefig(out_png)
        plt.close()
        print(f"  Gráfico salvo em: {out_png}")


if __name__ == "__main__":
    main()
