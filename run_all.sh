#!/bin/bash

# Use the src folder as base for imports
export PYTHONPATH=src

# ---------- CONFIGURAÇÕES ----------
SAMPLES=300
RUNS=10
# SEED=1751458615  # Comente para usar aleatória

# ---------- EXECUÇÃO DO PIPELINE ----------
echo "=== Rodando window_experiments.py ==="
if [ -z "$SEED" ]; then
    python3 -m word_embedding.window_experiments --samples $SAMPLES --runs $RUNS
else
    python3 -m word_embedding.window_experiments --samples $SAMPLES --runs $RUNS --seed $SEED
fi

echo "=== Calculando métricas ==="
python3 word_embedding/compute_partition_metrics.py

echo "=== Gerando heatmaps ==="
python3 word_embedding/plot_average_heatmap.py

echo "✅ Pipeline completo com $RUNS execuções e $SAMPLES amostras."
