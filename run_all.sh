#!/bin/bash

# Use the src folder as base for imports
export PYTHONPATH=src

# ---------- CONFIGURAÇÕES ----------
SAMPLES=300
RUNS=1
SEED=1752143761  # Comente para usar aleatória
N_BLOCKS=30      # Descomente para usar número fixo de blocos

# ---------- EXECUÇÃO DO PIPELINE ----------
echo "=== Rodando window_experiments.py ==="
ARGS="--samples $SAMPLES --runs $RUNS"

if [ -n "$SEED" ]; then
    ARGS="$ARGS --seed $SEED"
fi

if [ -n "$N_BLOCKS" ]; then
    ARGS="$ARGS --n_blocks $N_BLOCKS"
fi

python3 -m word_embedding.window_experiments $ARGS

echo "=== Calculando métricas ==="
python3 word_embedding/compute_partition_metrics.py

echo "=== Gerando heatmaps ==="
python3 word_embedding/plot_average_heatmap.py

echo "Pipeline completo com $RUNS execuções e $SAMPLES amostras."
