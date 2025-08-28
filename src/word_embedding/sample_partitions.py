#!/usr/bin/env python3
# sample_partitions.py — simplified for fixed schema:
# ['window','model','vertex','tipo','label','doc_id','term','label_members']

import argparse
import sys
import random
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Print partitions (terms) from a stored run.")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--model", default="sbm", choices=["sbm", "w2v"])
    ap.add_argument("--window", required=True)  # e.g. 5, 30, full
    ap.add_argument("--partition", type=str, help="single or comma-separated labels, e.g. 41 or 41,794")
    ap.add_argument("--n-groups", type=int, default=5)
    ap.add_argument("--random-seed", type=int)
    args = ap.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)

    # Build parquet path
    base = Path("../outputs/partitions")
    d = base / str(args.samples) / f"seed_{args.seed}" / f"{args.model}_J{args.window}"
    f = d / f"partitions_run{args.run:03d}.parquet"
    if not f.exists():
        print(f"[ERROR] Not found: {f}", file=sys.stderr)
        sys.exit(1)

    # Load parquet (fixed schema)
    df = pd.read_parquet(f)

    # Filter by model + window
    data = df[(df["model"] == args.model) & (df["window"].astype(str) == str(args.window))]
    if data.empty:
        print("No rows after filters.", file=sys.stderr); sys.exit(1)

    # Keep only term vertices (tipo == 1)
    terms_df = data[(data["tipo"] == 1) & (data["term"].notna())].copy()
    if terms_df.empty:
        print("No term vertices found.", file=sys.stderr); sys.exit(1)

    # label -> sorted unique terms
    groups = (
        terms_df.groupby("label")["term"]
        .apply(lambda s: sorted(set(map(str, s))))
        .to_dict()
    )
    if not groups:
        print("No partitions found.", file=sys.stderr); sys.exit(1)

    # Selection: specific partitions or random N
    if args.partition:
        wanted = []
        for tok in args.partition.split(","):
            tok = tok.strip()
            try:
                wanted.append(int(tok))
            except ValueError:
                print(f"[WARN] ignoring non-integer label '{tok}'", file=sys.stderr)
        selected = {lbl: groups[lbl] for lbl in wanted if lbl in groups}
        missing = [lbl for lbl in wanted if lbl not in groups]
        if missing:
            print(f"[WARN] Missing partitions: {missing}", file=sys.stderr)
        if not selected:
            print("No requested partitions found.", file=sys.stderr); sys.exit(1)
    else:
        labels = list(groups.keys())
        k = min(args.n_groups, len(labels))
        selected = {lbl: groups[lbl] for lbl in random.sample(labels, k)}

    # Print results
    print(f"=== MODEL: {args.model} | WINDOW: {args.window} | SAMPLES: {args.samples} | SEED: {args.seed} | RUN: {args.run} ===")
    for lbl, terms in sorted(selected.items(), key=lambda x: (-len(x[1]), x[0])):
        print(f"\nPartition label: {lbl}  |  #terms: {len(terms)}")
        print(", ".join(terms))

if __name__ == "__main__":
    main()
