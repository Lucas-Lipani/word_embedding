import json
from pathlib import Path
from datetime import datetime
from typing import Dict


TIPO_NAMES = {
    0: "Document",
    1: "Termo",
    2: "Cluster",
    3: "Janela",
    4: "Contexto",
    5: "JanelaSlide",
}


class ConfigManager:
    """
    Nova estrutura: conf/NNNN/config.json + run/RRRR/partition.parquet + results.json

    config.json: ENTRADA para UM MODELO ESPECÍFICO (SBM ou W2V)
    results.json: SAÍDA gerada durante execução (partições/clusters por tipo)
    """

    def __init__(self, base_conf_dir: Path):
        self.base_conf_dir = Path(base_conf_dir)
        self.base_conf_dir.mkdir(parents=True, exist_ok=True)

    def _get_corpus_signature(
        self,
        n_samples: int,
        seed: int,
        graph_type: str,
        nested: bool,
        layered: bool,
        n_blocks: int | None,
        window_size: int | str,
        edge_weighting: str = "uniform",
    ) -> str:
        """
        Assinatura baseada no corpus + window_size + seed + graph_type + nested + layered + n_blocks + edge_weighting.
        IDÊNTICA PARA AMBOS SBM E W2V!
        """
        return json.dumps(
            {
                "corpus": {
                    "seed": seed,
                    "number_of_documents": n_samples,
                },
                "graph": {
                    "graph_type": graph_type,
                    "sbm_variant": "nested" if nested else "flat",
                    "sbm_layered": layered,
                    "fixed_n_blocks": n_blocks,
                    "window_size": str(window_size),
                    "edge_weighting": edge_weighting,
                },
            },
            sort_keys=True,
        )

    def find_or_create_config_dirs(
        self,
        n_samples: int,
        seed: int,
        graph_type: str,
        nested: bool,
        layered: bool,
        n_blocks: int | None,
        window_size: int | str,
        edge_weighting: str = "uniform",
    ) -> tuple[Path, Path, int]:
        """
        Encontra/cria 2 configs: uma para SBM e outra para W2V.

        ASSINATURA: seed + samples + graph_type + nested + layered + n_blocks + window_size + edge_weighting

        :return: (config_dir_sbm, config_dir_w2v, corpus_signature_idx)
        """
        corpus_sig = self._get_corpus_signature(
            n_samples, seed, graph_type, nested, layered, n_blocks, window_size, edge_weighting
        )

        config_dirs = sorted(self.base_conf_dir.glob("????"))
        corpus_configs = {}  # Mapeia assinatura → {model_kind: config_dir}

        # Procurar configs existentes com mesma assinatura
        for config_dir in config_dirs:
            config_file = config_dir / "config.json"
            if not config_file.exists():
                print(f"[DEBUG] {config_dir.name}: sem config.json")
                continue

            try:
                with open(config_file, "r") as f:
                    saved_cfg = json.load(f)

                # Extrair assinatura do config salvo (MESMA LÓGICA)
                corpus = saved_cfg.get("corpus", {})
                graph = saved_cfg.get("graph", {})

                saved_sig = json.dumps(
                    {
                        "corpus": {
                            "seed": corpus.get("seed"),
                            "number_of_documents": corpus.get(
                                "number_of_documents"
                            ),
                        },
                        "graph": {
                            "graph_type": graph.get("graph_type"),
                            "sbm_variant": graph.get("sbm_variant"),
                            "sbm_layered": graph.get("sbm_layered", False),
                            "fixed_n_blocks": graph.get("fixed_n_blocks"),
                            "window_size": str(graph.get("window_size", "5")),
                        },
                    },
                    sort_keys=True,
                )

                if saved_sig == corpus_sig:
                    model_kind = saved_cfg.get("model", {}).get("kind")
                    if corpus_sig not in corpus_configs:
                        corpus_configs[corpus_sig] = {}
                    corpus_configs[corpus_sig][model_kind] = config_dir

            except Exception as e:
                print(f"[WARN] Erro ao ler {config_file}: {e}")

        # Se encontrou configs com mesma assinatura, VERIFICA SE AMBAS EXISTEM (SBM + W2V)
        if corpus_sig in corpus_configs:
            config_sbm = corpus_configs[corpus_sig].get("sbm")
            config_w2v = corpus_configs[corpus_sig].get("w2v+kmeans")

            # SÓ reutiliza se AMBAS existem
            if config_sbm and config_w2v:
                return config_sbm, config_w2v, int(config_sbm.name)

        # Senão, cria novos índices
        next_idx = max([int(d.name) for d in config_dirs], default=0) + 1
        config_dir_sbm = self.base_conf_dir / f"{next_idx:04d}"
        config_dir_w2v = self.base_conf_dir / f"{next_idx + 1:04d}"

        config_dir_sbm.mkdir(parents=True, exist_ok=True)
        config_dir_w2v.mkdir(parents=True, exist_ok=True)

        print(
            f"[CONFIG] ✗ Criando novos configs: SBM={config_dir_sbm.name}, W2V={config_dir_w2v.name}"
        )

        return config_dir_sbm, config_dir_w2v, next_idx

    def save_config(
        self,
        config_dir: Path,
        model_kind: str,
        n_samples: int,
        seed: int,
        graph_type: str,
        nested: bool,
        layered: bool,
        n_blocks: int | None,
        window_size: int | str,
        edge_weighting: str = "uniform",
    ) -> Path:
        """
        Salva config.json específico para SBM ou W2V.
        CONTÉM APENAS: ENTRADA (corpus + modelo específico + window_size + edge_weighting).
        """
        config_file = config_dir / "config.json"

        if config_file.exists():
            return config_file

        if model_kind == "sbm":
            cfg_data = {
                "timestamp": datetime.now().isoformat(),
                "corpus": {
                    "seed": seed,
                    "number_of_documents": n_samples,
                },
                "graph": {
                    "graph_type": graph_type,
                    "sbm_variant": "nested" if nested else "flat",
                    "sbm_layered": layered,
                    "fixed_n_blocks": n_blocks,
                    "window_size": window_size,
                    "edge_weighting": edge_weighting,
                },
                "model": {
                    "kind": "sbm",
                },
            }
        elif model_kind == "w2v":
            cfg_data = {
                "timestamp": datetime.now().isoformat(),
                "corpus": {
                    "seed": seed,
                    "number_of_documents": n_samples,
                },
                "graph": {
                    "graph_type": graph_type,
                    "sbm_variant": "nested" if nested else "flat",
                    "sbm_layered": layered,
                    "fixed_n_blocks": n_blocks,
                    "window_size": window_size,
                    "edge_weighting": edge_weighting,
                },
                "model": {
                    "kind": "w2v+kmeans",
                    "sg_algorithm": "skipgram",
                    "vector_size": 100,
                },
            }
        else:
            raise ValueError(f"model_kind desconhecido: {model_kind}")

        with open(config_file, "w") as f:
            json.dump(cfg_data, f, indent=2)

        print(f"[CONFIG] Metadados salvos em: {config_file}")
        return config_file

    def save_run_results(
        self,
        config_dir: Path,
        run_idx: int,
        model_kind: str,
        sbm_entropy: float | None = None,
        vertices_pre_sbm: Dict[int, int] = None,
        partitions_per_type: Dict[int, int] = None,
        w2v_n_clusters: int = None,
        clusters_per_type: Dict[int, int] = None,
        w2v_sg: int = None,
        w2v_window: int = None,
        w2v_vector_size: int = None,
    ) -> Path:
        """
        Salva results.json para um run específico.
        CONTÉM: SAÍDA gerada (partições/clusters por tipo) + CONFIGURAÇÃO do grafo.
        """
        run_dir = config_dir / "run" / f"{run_idx:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        results_file = run_dir / "results.json"

        # Ler configurações de config.json
        config_file = config_dir / "config.json"
        graph_config = {}
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    cfg_data = json.load(f)
                graph_info = cfg_data.get("graph", {})
                graph_config = {
                    "graph_type": graph_info.get("graph_type", "unknown"),
                    "sbm_variant": graph_info.get("sbm_variant", "flat"),
                    "sbm_layered": graph_info.get("sbm_layered", False),
                    "window_size": str(graph_info.get("window_size", "5")),
                }
            except Exception as e:
                print(f"[WARN] Erro ao ler config.json: {e}")

        # Converter tipos numéricos para nomes legíveis
        vertices_pre_sbm_named = {
            TIPO_NAMES.get(tipo, f"Tipo_{tipo}"): count
            for tipo, count in (vertices_pre_sbm or {}).items()
        }

        partitions_per_type_named = {
            TIPO_NAMES.get(tipo, f"Tipo_{tipo}"): count
            for tipo, count in (partitions_per_type or {}).items()
        }

        clusters_per_type_named = {
            TIPO_NAMES.get(tipo, f"Tipo_{tipo}"): count
            for tipo, count in (clusters_per_type or {}).items()
        }

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_kind,
            "graph_configuration": graph_config,
        }

        if model_kind == "sbm":
            results_data["sbm"] = {
                "entropy": sbm_entropy,
                "vertices_pre_sbm": vertices_pre_sbm_named,
                "total_vertices_pre_sbm": (
                    sum(vertices_pre_sbm.values()) if vertices_pre_sbm else 0
                ),
                "partitions_per_type": partitions_per_type_named,
            }
        elif model_kind == "w2v":
            results_data["w2v"] = {
                "number_of_clusters": w2v_n_clusters,
                "clusters_per_type": clusters_per_type_named,
                "sg_algorithm": "skipgram" if w2v_sg == 1 else "cbow",
                "context_window_size": w2v_window,
                "vector_size": w2v_vector_size,
            }

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"[RESULTS] Salvo em: {results_file}")
        return results_file
