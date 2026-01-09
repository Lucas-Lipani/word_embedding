import argparse
import sys
import random
from pathlib import Path
from collections import Counter
import pandas as pd
import spacy
import json


def tokenize_exact(nlp, text: str) -> list[str]:
    """Tokenização idêntica ao graph_build.py"""
    if not isinstance(text, str):
        return []
    doc = nlp(text)
    return [
        t.text.lower().strip() for t in doc if not t.is_stop and not t.is_punct
    ]


def recount_freqs_from_corpus(
    corpus_path: Path, text_col: str, samples: int, seed: int
) -> pd.DataFrame:
    """Recontagem de frequências do corpus original."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus não encontrado: {corpus_path}")

    if corpus_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(corpus_path)
    elif corpus_path.suffix.lower() == ".csv":
        df = pd.read_csv(corpus_path)
    else:
        raise ValueError(f"Formato não suportado: {corpus_path.suffix}")

    if text_col not in df.columns:
        raise ValueError(f"Coluna '{text_col}' não encontrada no corpus.")

    n_total = len(df)
    if samples > n_total:
        print(
            f"[WARN] samples={samples} > tamanho do corpus ({n_total}); usando {n_total}.",
            file=sys.stderr,
        )
        samples = n_total

    df_sel = df.sample(n=samples, random_state=int(seed), replace=False)
    nlp = spacy.load("en_core_web_sm")

    counter = Counter()
    for txt in df_sel[text_col].fillna(""):
        counter.update(tokenize_exact(nlp, txt))

    return pd.DataFrame(counter.items(), columns=["term", "freq"])


def find_matching_partition(
    model: str,
    window: str,
    config: int | None = None,
    run: int | None = None,
    seed: str | None = None,
) -> tuple[Path, Path, str, str] | None:
    """
    Procura por partition.parquet que corresponda aos critérios.

    Se config+run são fornecidos:
      - Procura especificamente naquele run

    Se apenas model+window são fornecidos (sem config/run):
      - Varre todas as configs buscando model+window
      - Se seed é fornecido, filtra por seed
      - Se múltiplas configs encontram, retorna lista e pede confirmação

    :return: (config_dir, partition_file, model, window) ou None se não encontrado
    """
    base = Path("../outputs/conf")

    if not base.exists():
        print(
            f"[ERROR] Diretório base não encontrado: {base}", file=sys.stderr
        )
        return None

    # CASO 1: config + run especificados (busca específica)
    if config is not None and run is not None:
        config_dir = base / f"{config:04d}"

        if not config_dir.exists():
            print(
                f"[ERROR] Config não encontrada: {config_dir}", file=sys.stderr
            )
            available = sorted([d.name for d in base.glob("????")])
            if available:
                print(
                    f"[HINT] Configs disponíveis: {', '.join(available)}",
                    file=sys.stderr,
                )
            return None

        run_dir = config_dir / "run" / f"{run:04d}"
        partition_file = run_dir / "partition.parquet"

        if not partition_file.exists():
            print(
                f"[ERROR] Partition não encontrado: {partition_file}",
                file=sys.stderr,
            )
            available_runs = sorted(
                [d.name for d in (config_dir / "run").glob("????")]
            )
            if available_runs:
                print(
                    f"[HINT] Runs disponíveis: {', '.join(available_runs)}",
                    file=sys.stderr,
                )
            return None

        # Verificar se model+window estão no parquet
        df = pd.read_parquet(partition_file)
        has_model_window = not df[
            (df["model"] == model) & (df["window"].astype(str) == str(window))
        ].empty

        if not has_model_window:
            print(
                f"[ERROR] Model={model}, window={window} não encontrado em config {config:04d}, run {run:04d}",
                file=sys.stderr,
            )
            available_models = df["model"].unique().tolist()
            available_windows = df["window"].unique().tolist()
            print(
                f"[HINT] Modelos disponíveis: {available_models}",
                file=sys.stderr,
            )
            print(
                f"[HINT] Janelas disponíveis: {available_windows}",
                file=sys.stderr,
            )
            return None

        return (config_dir, partition_file, model, str(window))

    # CASO 2: apenas model+window (busca genérica)
    print(f"\n[SEARCH] Procurando model={model}, window={window}")

    matching_configs = []

    for config_dir in sorted(base.glob("????")):
        if not config_dir.is_dir():
            continue

        # Carregar config.json para extrair seed
        config_file = config_dir / "config.json"
        if not config_file.exists():
            continue

        try:
            with open(config_file, "r") as f:
                cfg = json.load(f)
                config_seed = cfg.get("corpus", {}).get("seed")
        except Exception:
            continue

        # Se seed foi especificada, filtrar
        if seed is not None:
            try:
                if int(config_seed) != int(seed):
                    continue
            except (ValueError, TypeError):
                continue

        # Procurar runs com model+window
        run_dirs = sorted((config_dir / "run").glob("????"))

        for run_dir in run_dirs:
            partition_file = run_dir / "partition.parquet"
            if not partition_file.exists():
                continue

            try:
                df = pd.read_parquet(partition_file)
                has_model_window = not df[
                    (df["model"] == model)
                    & (df["window"].astype(str) == str(window))
                ].empty

                if has_model_window:
                    run_idx = int(run_dir.name)
                    config_idx = int(config_dir.name)
                    matching_configs.append(
                        {
                            "config": config_idx,
                            "run": run_idx,
                            "seed": config_seed,
                            "partition_file": partition_file,
                            "config_dir": config_dir,
                        }
                    )
            except Exception:
                continue

    if not matching_configs:
        print(
            f"[ERROR] Nenhuma partição encontrada para model={model}, window={window}",
            file=sys.stderr,
        )
        if seed is not None:
            print(f"[HINT] Com seed={seed}", file=sys.stderr)
        return None

    # Se encontrou exatamente uma, retorna
    if len(matching_configs) == 1:
        match = matching_configs[0]
        print(
            f"[FOUND] Config {match['config']:04d}, run {match['run']:04d}, seed={match['seed']}"
        )
        return (
            match["config_dir"],
            match["partition_file"],
            model,
            str(window),
        )

    # Se encontrou múltiplas, agrupa por seed e pede confirmação
    by_seed = {}
    for match in matching_configs:
        s = match["seed"]
        if s not in by_seed:
            by_seed[s] = []
        by_seed[s].append(match)

    print(f"\n[FOUND] {len(matching_configs)} partições encontradas:")
    for s in sorted(by_seed.keys()):
        matches = by_seed[s]
        print(f"\n  Seed {s}:")
        for m in matches:
            print(f"    - config {m['config']:04d}, run {m['run']:04d}")

    if seed is None:
        print(
            f"\n[HINT] Especifique --seed para filtrar, ou use --config + --run para escolher uma específica",
            file=sys.stderr,
        )
        return None

    # Retornar a primeira encontrada com seed especificado
    match = matching_configs[0]
    print(f"\n[SELECT] Config {match['config']:04d}, run {match['run']:04d}")
    return (match["config_dir"], match["partition_file"], model, str(window))


def main():
    ap = argparse.ArgumentParser(
        description="Inspeciona partições com busca flexível (NOVA ESTRUTURA)."
    )
    ap.add_argument(
        "--model",
        default="sbm",
        choices=["sbm", "w2v"],
        help="Modelo a inspecionar",
    )
    ap.add_argument("--window", required=True, help="Janela (ex: 5, 20, full)")
    ap.add_argument(
        "--config",
        "-c",
        type=int,
        default=None,
        help="[OPCIONAL] Config específica (ex: 1)",
    )
    ap.add_argument(
        "--run",
        type=int,
        default=None,
        help="[OPCIONAL] Run específico (ex: 1)",
    )
    ap.add_argument(
        "--seed",
        "-s",
        type=str,
        default=None,
        help="[OPCIONAL] Seed para filtrar (ex: 42)",
    )
    ap.add_argument(
        "--print-k", type=int, default=5, help="Quantas partições imprimir"
    )
    ap.add_argument(
        "--corpus",
        default="../wos_sts_journals.parquet",
        help="Caminho do corpus",
    )
    ap.add_argument(
        "--text-col", default="abstract", help="Coluna de texto no corpus"
    )
    ap.add_argument(
        "--random-seed", type=int, help="Seed para amostrar partições"
    )
    ap.add_argument(
        "--samples", type=int, default=100, help="Número de samples do corpus"
    )
    args = ap.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)

    # Buscar partição
    result = find_matching_partition(
        model=args.model,
        window=args.window,
        config=args.config,
        run=args.run,
        seed=args.seed,
    )

    if result is None:
        sys.exit(1)

    config_dir, partition_file, model, window = result

    # Carregar parquet
    try:
        df = pd.read_parquet(partition_file)
    except Exception as e:
        print(f"[ERROR] Falha ao carregar parquet: {e}", file=sys.stderr)
        sys.exit(1)

    # Filtrar por model + window
    data = df[
        (df["model"] == model) & (df["window"].astype(str) == str(window))
    ]

    if data.empty:
        print(
            f"[ERROR] Nenhum dado encontrado para model={model}, window={window}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Filtrar termos
    if "tipo" in data.columns:
        terms_df = data[(data["term"].notna()) & (data["tipo"] == 1)].copy()
        if terms_df.empty:
            terms_df = data[data["term"].notna()].copy()
    else:
        terms_df = data[data["term"].notna()].copy()

    if terms_df.empty:
        print("[ERROR] Sem termos neste parquet.", file=sys.stderr)
        sys.exit(1)

    # Recontar frequências do corpus
    try:
        tf = recount_freqs_from_corpus(
            Path(args.corpus), args.text_col, args.samples, 42
        )
    except Exception as e:
        print(f"[ERROR] Falha ao contar frequências: {e}", file=sys.stderr)
        sys.exit(1)

    tf = tf[tf["term"].isin(terms_df["term"].unique())]
    merged = terms_df.merge(tf, on="term", how="left").fillna({"freq": 0})
    merged["freq"] = merged["freq"].astype(int)

    labels = merged["label"].unique().tolist()
    if not labels:
        print("[ERROR] Nenhuma partição encontrada.", file=sys.stderr)
        sys.exit(1)

    # Salvar TXT com TODAS as partições
    run_idx = int(partition_file.parent.name)
    config_idx = int(config_dir.name)

    out_txt = (
        partition_file.parent / f"terms_by_frequency_run{run_idx:04d}.txt"
    )
    part_stats = (
        merged.groupby("label")["freq"]
        .agg(total_term_tokens="sum", distinct_terms="count")
        .reset_index()
        .sort_values(["total_term_tokens", "label"], ascending=[False, True])
    )
    order = part_stats["label"].tolist()

    with open(out_txt, "w", encoding="utf-8") as w:
        header = f"MODEL={model} | WINDOW={window} | CONFIG={config_idx:04d} | RUN={run_idx:04d}"
        w.write(header + "\n" + "=" * len(header) + "\n\n")

        for lbl in order:
            sub = merged[merged["label"] == lbl].sort_values(
                ["freq", "term"], ascending=[False, True]
            )
            total_tokens = int(sub["freq"].sum())
            distinct = int(sub.shape[0])
            w.write(
                f"Partition {lbl}  |  terms: {distinct}  |  tokens: {total_tokens}\n"
            )
            for _, row in sub.iterrows():
                w.write(f"{row['term']}\t{int(row['freq'])}\n")
            w.write("\n")

    # Imprimir K partições escolhidas
    k = min(args.print_k, len(labels))
    chosen = sorted(random.sample(labels, k))

    print(
        f"\n=== MODEL: {model} | WINDOW: {window} | CONFIG: {config_idx:04d} | RUN: {run_idx:04d} ==="
    )
    print(f"[SAVED] {out_txt}\n")

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
            f"Partition {lbl}  |  terms: {distinct}  |  tokens: {total_tokens}"
        )
        print(preview + "\n")


if __name__ == "__main__":
    main()
