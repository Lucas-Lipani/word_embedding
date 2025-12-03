import argparse
import sys
import random
from pathlib import Path
from collections import Counter
import pandas as pd
import spacy


# ========= tokenização idêntica ao graph_build.py =========
def tokenize_exact(nlp, text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    doc = nlp(text)
    return [
        t.text.lower().strip() for t in doc if not t.is_stop and not t.is_punct
    ]


def recount_freqs_from_corpus(
    corpus_path: Path, text_col: str, samples: int, seed: int
) -> pd.DataFrame:
    # lê corpus e seleciona exatamente o mesmo subset usado na run (amostragem determinística)
    if corpus_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(corpus_path)
    elif corpus_path.suffix.lower() == ".csv":
        df = pd.read_csv(corpus_path)
    else:
        raise ValueError(
            f"Formato não suportado para corpus: {corpus_path.suffix}"
        )
    if text_col not in df.columns:
        raise ValueError(f"Coluna '{text_col}' não encontrada no corpus.")

    n_total = len(df)
    if samples > n_total:
        print(
            f"[WARN] samples={samples} > tamanho do corpus ({n_total}); usando {n_total}.",
            file=sys.stderr,
        )
        samples = n_total

    # amostragem determinística igual à pipeline
    df_sel = df.sample(n=samples, random_state=int(seed), replace=False)
    nlp = spacy.load("en_core_web_sm")

    counter = Counter()
    for txt in df_sel[text_col].fillna(""):
        counter.update(tokenize_exact(nlp, txt))

    return pd.DataFrame(counter.items(), columns=["term", "freq"])


def main():
    ap = argparse.ArgumentParser(
        description="Imprime 5 partições aleatórias ordenadas por frequência (recontada do corpus) e salva um TXT com todas."
    )
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument(
        "--config",
        type=int,
        default=1,
        help="Número da CONFIG (ex: 1 para config_001)",
    )
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--model", default="sbm", choices=["sbm", "w2v"])
    ap.add_argument("--window", required=True)  # ex.: 5, 30, full
    ap.add_argument(
        "--print-k",
        type=int,
        default=5,
        help="quantas partições imprimir em tela (default: 5)",
    )
    # corpus (padrões do seu projeto)
    ap.add_argument(
        "--corpus",
        default="../wos_sts_journals.parquet",
        help="caminho do corpus (Parquet/CSV)",
    )
    ap.add_argument(
        "--text-col",
        default="abstract",
        help="coluna de texto no corpus (default: abstract)",
    )
    ap.add_argument(
        "--random-seed",
        type=int,
        help="seed apenas para amostrar as partições que serão impressas",
    )
    args = ap.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)

    # caminho do parquet de partição (NOVA ESTRUTURA COM CONFIG)
    base = Path("../outputs/partitions")
    d = (
        base
        / str(args.samples)
        / f"seed_{args.seed}"
        / f"config_{args.config:03d}"
        / f"{args.model}_J{args.window}"
    )
    parquet_path = d / f"partitions_run{args.run:03d}.parquet"

    if not parquet_path.exists():
        print(
            f"[ERROR] parquet não encontrado: {parquet_path}", file=sys.stderr
        )
        sys.exit(1)

    # lê partições
    df = pd.read_parquet(parquet_path)
    data = df[
        (df["model"] == args.model)
        & (df["window"].astype(str) == str(args.window))
    ]
    for col in ("term", "label"):
        if col not in data.columns:
            print(f"[ERROR] coluna ausente no parquet: {col}", file=sys.stderr)
            sys.exit(1)

    # suporta com/sem coluna 'tipo'
    if "tipo" in data.columns:
        terms_df = data[(data["term"].notna()) & (data["tipo"] == 1)].copy()
        if terms_df.empty:
            # fallback: se não tiver tipo==1, usa todos com term
            terms_df = data[data["term"].notna()].copy()
    else:
        terms_df = data[data["term"].notna()].copy()

    if terms_df.empty:
        print("[ERROR] não há termos nesta run após filtro.", file=sys.stderr)
        sys.exit(1)

    # reconta frequências do corpus original (subset samples+seed)
    tf = recount_freqs_from_corpus(
        Path(args.corpus), args.text_col, int(args.samples), int(args.seed)
    )

    # mantém só termos presentes nas partições e faz merge
    tf = tf[tf["term"].isin(terms_df["term"].unique())]
    merged = terms_df.merge(tf, on="term", how="left").fillna({"freq": 0})
    merged["freq"] = merged["freq"].astype(int)

    labels = merged["label"].unique().tolist()
    if not labels:
        print("[ERROR] nenhuma partição encontrada.", file=sys.stderr)
        sys.exit(1)

    # escolhe K partições aleatórias para imprimir
    k = min(args.print_k, len(labels))
    chosen = sorted(random.sample(labels, k))

    # salva TXT com TODAS as partições, ordenadas por total de tokens (desc)
    out_txt = d / f"terms_by_frequency_run{args.run:03d}.txt"
    part_stats = (
        merged.groupby("label")["freq"]
        .agg(total_term_tokens="sum", distinct_terms="count")
        .reset_index()
        .sort_values(["total_term_tokens", "label"], ascending=[False, True])
    )
    order = part_stats["label"].tolist()

    with open(out_txt, "w", encoding="utf-8") as w:
        header = f"MODEL={args.model} | WINDOW={args.window} | SAMPLES={args.samples} | SEED={args.seed} | CONFIG={args.config:03d} | RUN={args.run}"
        w.write(header + "\n")
        w.write("=" * len(header) + "\n\n")
        for lbl in order:
            sub = merged[merged["label"] == lbl].sort_values(
                ["freq", "term"], ascending=[False, True]
            )
            total_tokens = int(sub["freq"].sum())
            distinct = int(sub.shape[0])
            w.write(
                f"Partition label: {lbl}  |  #distinct_terms: {distinct}  |  total_term_tokens: {total_tokens}\n"
            )
            for _, row in sub.iterrows():
                w.write(f"{row['term']}\t{int(row['freq'])}\n")
            w.write("\n")

    # imprime as K partições escolhidas
    print(
        f"=== MODEL: {args.model} | WINDOW: {args.window} | SAMPLES: {args.samples} | SEED: {args.seed} | CONFIG: {args.config:03d} | RUN: {args.run} ==="
    )
    print(f"[saved] {out_txt}")
    for lbl in chosen:
        sub = merged[merged["label"] == lbl].sort_values(
            ["freq", "term"], ascending=[False, True]
        )
        total_tokens = int(sub["freq"].sum())
        distinct = int(sub.shape[0])
        preview = ", ".join(
            f"{t} ({int(c)})" for t, c in zip(sub["term"], sub["freq"])
        )
        print(
            f"\nPartition label: {lbl}  |  #distinct_terms: {distinct}  |  total_term_tokens: {total_tokens}"
        )
        print(preview)


if __name__ == "__main__":
    main()
