import json
from pathlib import Path
from datetime import datetime
from typing import List


class AnalysisManager:
    """
    Gerencia a estrutura: analyses/NNNN/config.json + results.json
    
    config.json: ENTRADA - quais configs foram comparadas
    results.json: SAÍDA - métricas aggregadas da análise
    """

    def __init__(self, base_analyses_dir: Path):
        self.base_analyses_dir = Path(base_analyses_dir)
        self.base_analyses_dir.mkdir(parents=True, exist_ok=True)

    def find_or_create_analysis_dir(
        self,
        config_indices: List[int],
        comparison_type: str,  # "sbm_vs_w2v", "sbm_vs_sbm", "w2v_vs_w2v"
    ) -> tuple[Path, int]:
        """
        Encontra/cria pasta analyses/NNNN para uma análise específica.
        
        Assinatura: sorted(config_indices) + comparison_type
        
        :param config_indices: Lista de índices de configs a comparar (ex: [1, 2])
        :param comparison_type: Tipo de comparação
        :return: (analysis_dir, analysis_idx)
        """
        # >>> CORRIGIDO: converter para int (remove numpy.int64)
        config_indices_sorted = sorted([int(idx) for idx in config_indices])
        
        analysis_sig = json.dumps(
            {
                "configs": config_indices_sorted,
                "comparison": comparison_type,
            },
            sort_keys=True,
        )

        print(f"[DEBUG] Analysis signature: {analysis_sig}")

        analysis_dirs = sorted(self.base_analyses_dir.glob("????"))
        
        # Procurar análises existentes com mesma assinatura
        for analysis_dir in analysis_dirs:
            config_file = analysis_dir / "config.json"
            if not config_file.exists():
                continue

            try:
                with open(config_file, "r") as f:
                    saved_cfg = json.load(f)

                saved_sig = json.dumps(
                    {
                        "configs": sorted(saved_cfg.get("configs", [])),
                        "comparison": saved_cfg.get("comparison_type", ""),
                    },
                    sort_keys=True,
                )

                if saved_sig == analysis_sig:
                    idx = int(analysis_dir.name)
                    print(
                        f"[ANALYSIS] ✓ Reutilizando análise existente: {analysis_dir.name}"
                    )
                    return analysis_dir, idx
            except Exception as e:
                print(f"[WARN] Erro ao ler {config_file}: {e}")

        # Criar nova análise
        next_idx = (
            max([int(d.name) for d in analysis_dirs], default=0) + 1
            if analysis_dirs
            else 1
        )
        analysis_dir = self.base_analyses_dir / f"{next_idx:04d}"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[ANALYSIS] ✗ Criando nova análise: {analysis_dir.name}"
        )

        return analysis_dir, next_idx

    def save_analysis_config(
        self,
        analysis_dir: Path,
        config_indices: List[int],
        comparison_type: str,
        corpus_seed: int | None = None,
        n_samples: int | None = None,
        graph_type: str | None = None,
    ) -> Path:
        """
        Salva config.json da análise.
        ENTRADA: quais configs foram comparadas e metadados do corpus.
        """
        config_file = analysis_dir / "config.json"

        if config_file.exists():
            print(
                f"[ANALYSIS] config.json já existe em {analysis_dir.name}, preservando..."
            )
            return config_file

        # >>> CORRIGIDO: converter para int
        config_indices = [int(idx) for idx in config_indices]

        cfg_data = {
            "timestamp": datetime.now().isoformat(),
            "configs_compared": sorted(config_indices),
            "comparison_type": comparison_type,
            "corpus": {
                "seed": corpus_seed,
                "number_of_documents": n_samples,
            },
            "graph": {
                "graph_type": graph_type,
            },
        }

        with open(config_file, "w") as f:
            json.dump(cfg_data, f, indent=2)

        print(f"[ANALYSIS] config.json salvo em: {config_file}")
        return config_file

    def save_analysis_results(
        self,
        analysis_dir: Path,
        results_df,  # pandas DataFrame com métricas
        comparison_type: str,
    ) -> Path:
        """
        Salva results.json com agregações de métricas.
        SAÍDA: resumo das análises (NMI, VI, ARI médios, etc).
        """
        results_file = analysis_dir / "results.json"

        # Calcular estatísticas agregadas
        metrics = {}
        for col in results_df.columns:
            if col not in {"config_x", "config_y", "run_x", "run_y", "window_x", "window_y"}:
                metrics[col] = {
                    "mean": float(results_df[col].mean()),
                    "std": float(results_df[col].std()),
                    "min": float(results_df[col].min()),
                    "max": float(results_df[col].max()),
                }

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "comparison_type": comparison_type,
            "total_comparisons": int(len(results_df)),
            "metrics_summary": metrics,
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"[ANALYSIS] results.json salvo em: {results_file}")
        return results_file
