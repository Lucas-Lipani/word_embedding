from pathlib import Path
import pandas as pd
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
                n_clusters = df["label"].nunique()
                rows.append({
                    "model": model,
                    "window": window,
                    "run": run,
                    "n_clusters": n_clusters
                })

        if rows:
            df = pd.DataFrame(rows)
            df.sort_values(["model", "window", "run"], inplace=True)

            analysis_dir = seed_dir / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)

            output_file = analysis_dir / "cluster_counts.csv"
            df.to_csv(output_file, index=False)
            print(f"  Arquivo salvo em: {output_file}")
        else:
            print(f"  Nenhum dado encontrado para {seed_dir}")
