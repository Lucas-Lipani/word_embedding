import argparse
import sys
import random
import pandas as pd
from pathlib import Path


def detect_col(df, candidates):
    """Return the first existing column name among candidates, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Print N random SBM partitions (terms) from project outputs."
    )
    parser.add_argument(
        "--samples", type=str, required=True, help="Sample size (e.g. 1200)"
    )
    parser.add_argument(
        "--seed",
        type=str,
        required=True,
        help="Seed identifier (e.g. 1755724084)",
    )
    parser.add_argument(
        "--run", type=int, required=True, help="Run number (e.g. 1)"
    )
    parser.add_argument(
        "--model", default="sbm", help="Model to filter (default: sbm)"
    )
    parser.add_argument(
        "--window",
        required=True,
        help="Window size to filter (e.g., 5, 10, full)",
    )

    parser.add_argument(
        "--partition",
        type=int,
        help="Specific partition label to print (overrides random sampling).",
    )

    parser.add_argument(
        "--n-groups",
        type=int,
        default=5,
        help="How many random partitions to print (default: 5).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Set RNG seed for reproducibility (optional).",
    )

    args = parser.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)

    # Constroi o caminho do arquivo parquet
    base = Path("../outputs/partitions")
    seed_dir = (
        base
        / str(args.samples)
        / f"seed_{args.seed}"
        / f"{args.model}_J{args.window}"
    )
    parquet_file = seed_dir / f"partitions_run{args.run:03d}.parquet"

    if not parquet_file.exists():
        print(f"[ERROR] File not found: {parquet_file}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"Error reading parquet: {e}", file=sys.stderr)
        sys.exit(1)

    # Detecção de colunas
    col_model = detect_col(df, ["model"])
    col_window = detect_col(df, ["window"])
    col_tipo = detect_col(df, ["tipo", "type"])
    col_term = detect_col(df, ["term", "token", "word"])
    col_label = detect_col(df, ["label", "partition", "block"])

    needed = [col_model, col_window, col_tipo, col_term, col_label]
    if any(c is None for c in needed):
        print("Parquet is missing required columns", file=sys.stderr)
        sys.exit(1)

    # Filtra por modelo + janela
    data = df.copy()
    data = data[data[col_model] == args.model]
    data = data[data[col_window].astype(str) == str(args.window)]

    if data.empty:
        print("No rows after filters.", file=sys.stderr)
        sys.exit(1)

    # Mantém apenas vértices do tipo termo (tipo == 1)
    term_mask = data[col_tipo] == 1
    if data[col_tipo].dtype == object:
        term_mask = data[col_tipo].astype(str) == "1"

    terms_df = data[term_mask & data[col_term].notna()].copy()
    if terms_df.empty:
        print("No term vertices found.", file=sys.stderr)
        sys.exit(1)

    # Constroi o mapeamento partição -> termos
    groups = (
        terms_df.groupby(col_label)[col_term]
        .apply(lambda s: sorted(set(map(str, s))))
        .to_dict()
    )

    if not groups:
        print("No partitions found.", file=sys.stderr)
        sys.exit(1)

    # Se uma partição específica for solicitada, mostre apenas essa
    if args.partition is not None:
        lbl = args.partition
        if lbl not in groups:
            keys_as_int = {
                int(k): k for k in groups.keys() if str(k).isdigit()
            }
            key = keys_as_int.get(lbl, None)
            if key is None:
                print(f"Partition {lbl} not found.", file=sys.stderr)
                sys.exit(1)
            lbl = key
        selected = {lbl: groups[lbl]}
    else:
        labels = list(groups.keys())
        k = min(args.n_groups, len(labels))
        selected_labels = random.sample(labels, k)
        selected = {lbl: groups[lbl] for lbl in selected_labels}

    # Print saída no terminal
    print(
        f"=== MODEL: {args.model} | WINDOW: {args.window} | SAMPLES: {args.samples} | SEED: {args.seed} | RUN: {args.run} ==="
    )

    for lbl, terms in sorted(
        selected.items(), key=lambda x: (len(x[1]) * -1, x[0])
    ):
        print(f"\nPartition label: {lbl}  |  #terms: {len(terms)}")
        print(", ".join(terms))


if __name__ == "__main__":
    main()
