import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


# Mapeamento de tipos numéricos para nomes legíveis
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
    Gerencia estrutura: conf/NNNN/config.json + run/NNNN/partition.parquet + parameters.json
    """

    def __init__(self, base_conf_dir: Path):
        """
        :param base_conf_dir: Caminho da pasta conf (ex: outputs/conf)
        """
        self.base_conf_dir = Path(base_conf_dir)
        self.base_conf_dir.mkdir(parents=True, exist_ok=True)

    def _get_config_signature(
        self,
        n_samples: int,
        seed: int,
        graph_type: str,
        nested: bool,
        n_blocks: int | None,
    ) -> str:
        """
        Cria uma assinatura única para uma configuração.
        NOTA: Ignora timestamp para que configs idênticas (exceto timestamp) sejam reutilizadas.
        Contém APENAS os campos relevantes (corpus + model, SEM kind/variant extras).
        """
        return json.dumps(
            {
                "corpus": {
                    "seed": seed,
                    "number_of_documents": n_samples,
                },
                "model": {
                    "graph_type": graph_type,
                    "sbm_variant": "nested" if nested else "flat",
                    "fixed_n_blocks": n_blocks,
                },
            },
            sort_keys=True,
        )

    def find_or_create_config_dir(
        self,
        n_samples: int,
        seed: int,
        graph_type: str,
        nested: bool,
        n_blocks: int | None,
    ) -> tuple[Path, int, bool]:
        """
        Procura uma pasta conf/NNNN que já tenha a mesma assinatura.
        Se encontrar, retorna seu caminho e índice (reutiliza=True).
        Se não encontrar, cria uma nova com índice sequencial (reutiliza=False).

        :return: (caminho da pasta conf/NNNN, índice, foi_reutilizada)
        """
        target_sig = self._get_config_signature(
            n_samples, seed, graph_type, nested, n_blocks
        )

        print(f"[DEBUG] Target signature: {target_sig}")

        # Procura pastas conf_* existentes
        config_dirs = sorted(self.base_conf_dir.glob("????"))

        for config_dir in config_dirs:
            config_file = config_dir / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        saved_cfg = json.load(f)

                    # Criar assinatura do config salvo (SEM timestamp, APENAS corpus+model)
                    # IMPORTANTE: extrair APENAS os campos que usamos na assinatura alvo
                    saved_sig = json.dumps(
                        {
                            "corpus": saved_cfg.get("corpus", {}),
                            "model": {
                                "graph_type": saved_cfg.get("model", {}).get(
                                    "graph_type"
                                ),
                                "sbm_variant": saved_cfg.get("model", {}).get(
                                    "variant"
                                ),
                                "fixed_n_blocks": saved_cfg.get(
                                    "model", {}
                                ).get("fixed_n_blocks"),
                            },
                        },
                        sort_keys=True,
                    )

                    print(f"[DEBUG] Checking {config_dir.name}: {saved_sig}")

                    # Comparar assinaturas (ignorando timestamp e campos extras como 'kind')
                    if saved_sig == target_sig:
                        idx = int(config_dir.name)
                        print(
                            f"[CONFIG] ✓ Reutilizando pasta existente: {config_dir.name}"
                        )
                        return config_dir, idx, True
                except Exception as e:
                    print(f"[WARN] Erro ao ler {config_file}: {e}")

        # Se não encontrou, cria nova pasta com índice sequencial
        next_idx = (
            max([int(d.name) for d in config_dirs], default=0) + 1
            if config_dirs
            else 1
        )
        new_config_dir = self.base_conf_dir / f"{next_idx:04d}"
        new_config_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[CONFIG] ✗ Criando nova pasta de configuração: {new_config_dir.name}"
        )

        return new_config_dir, next_idx, False

    def save_config(
        self,
        config_dir: Path,
        n_samples: int,
        seed: int,
        graph_type: str,
        nested: bool,
        n_blocks: int | None,
    ) -> Path:
        """
        Salva config.json na pasta config_dir (apenas se não existir).
        Se já existe, não sobrescreve (preserva timestamp original).

        :return: Caminho do config.json salvo
        """
        config_file = config_dir / "config.json"

        # Se já existe, não sobrescreve
        if config_file.exists():
            print(
                f"[CONFIG] config.json já existe em {config_dir.name}, preservando..."
            )
            return config_file

        cfg_data = {
            "timestamp": datetime.now().isoformat(),
            "corpus": {
                "seed": seed,
                "number_of_documents": n_samples,
            },
            "model": {
                "kind": "sbm",
                "variant": "nested" if nested else "flat",
                "graph_type": graph_type,
                "fixed_n_blocks": n_blocks,
            },
        }

        with open(config_file, "w") as f:
            json.dump(cfg_data, f, indent=2)

        print(f"[CONFIG] Metadados salvos em: {config_file}")
        return config_file

    def save_run_parameters(
        self,
        config_dir: Path,
        run_idx: int,
        sbm_entropy: float | None = None,
        # >>> NOVO: informações detalhadas do grafo e SBM
        vertices_pre_sbm: Dict[int, int] = None,
        blocks_post_sbm: Dict[int, int] = None,
        term_blocks_count: int = None,
        window_blocks_count: int = None,
        # >>> NOVO: informações do W2V
        w2v_n_clusters: int = None,
        w2v_sg: int = None,
        w2v_window: int = None,
        w2v_vector_size: int = None,
    ) -> Path:
        """
        Salva parameters.json para um run específico com informações detalhadas.

        :param config_dir: Pasta config/NNNN
        :param run_idx: Índice da execução (0001, 0002, ...)
        :param sbm_entropy: Entropy do SBM
        :param vertices_pre_sbm: Dict {tipo: count} - estrutura do grafo ANTES do SBM
        :param blocks_post_sbm: Dict {tipo: count} - número de blocos APÓS SBM por tipo
        :param term_blocks_count: Número de blocos que contêm TERMOS
        :param window_blocks_count: Número de blocos que contêm JANELAS
        :param w2v_n_clusters: Número de clusters do K-Means (= term_blocks_count)
        :param w2v_sg: 0 para CBOW, 1 para Skip-gram
        :param w2v_window: Tamanho da janela do W2V
        :param w2v_vector_size: Dimensionalidade dos vetores
        :return: Caminho do parameters.json
        """
        run_dir = config_dir / "run" / f"{run_idx:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        parameters_file = run_dir / "parameters.json"

        # Converter tipos numéricos para nomes legíveis
        vertices_pre_sbm_named = {
            TIPO_NAMES.get(tipo, f"Tipo_{tipo}"): count
            for tipo, count in (vertices_pre_sbm or {}).items()
        }

        blocks_post_sbm_named = {
            TIPO_NAMES.get(tipo, f"Tipo_{tipo}"): count
            for tipo, count in (blocks_post_sbm or {}).items()
        }

        # >>> NOVO: Estrutura W2V
        w2v_model = {
            "kind": "w2v+kmeans",
            "number_of_clusters": w2v_n_clusters,
            "sg_algorithm": "skipgram" if w2v_sg == 1 else "cbow",
            "window_size": w2v_window,
            "vector_size": w2v_vector_size,
        }

        params_data = {
            "timestamp": datetime.now().isoformat(),
            "entropy": sbm_entropy,
            "graph_structure": {
                "vertices_pre_sbm": vertices_pre_sbm_named,
                "total_vertices_pre_sbm": (
                    sum(vertices_pre_sbm.values()) if vertices_pre_sbm else 0
                ),
            },
            "sbm_results": {
                "blocks_post_sbm": blocks_post_sbm_named,
                "term_blocks_count": term_blocks_count,
                "window_blocks_count": window_blocks_count,
            },
            "w2v_model": w2v_model,  # ← NOVO
        }

        with open(parameters_file, "w") as f:
            json.dump(params_data, f, indent=2)

        print(f"[PARAMS] Salvo em: {parameters_file}")
        return parameters_file

    def save_derived_metrics(
        self, config_dir: Path, derived_metrics: Dict[str, Any]
    ) -> Path:
        """
        Salva derived.json com médias de métricas agregadas.

        :param config_dir: Pasta config/NNNN
        :param derived_metrics: Dict com médias NMI, VI, etc.
        :return: Caminho do derived.json
        """
        derived_file = config_dir / "derived.json"

        derived_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": derived_metrics,
        }

        with open(derived_file, "w") as f:
            json.dump(derived_data, f, indent=2)

        print(f"[DERIVED] Salvo em: {derived_file}")
        return derived_file


def print_config_info(config_dir: Path):
    """
    Imprime informações da configuração.
    """
    config_file = config_dir / "config.json"
    if not config_file.exists():
        print(f"[INFO] Nenhuma configuração encontrada em {config_dir}")
        return

    with open(config_file, "r") as f:
        cfg = json.load(f)

    print(f"\n{'='*70}")
    print(f"Pasta de configuração: {config_dir.name}")
    print(f"{'='*70}")
    print(f"Timestamp:        {cfg.get('timestamp', 'N/A')}")
    print(f"\nCorpus:")
    corpus = cfg.get("corpus", {})
    print(f"  Seed:           {corpus.get('seed', 'N/A')}")
    print(f"  Documents:      {corpus.get('number_of_documents', 'N/A')}")
    print(f"\nModel:")
    model = cfg.get("model", {})
    print(f"  Kind:           {model.get('kind', 'N/A')}")
    print(f"  Variant:        {model.get('variant', 'N/A')}")
    print(f"  Graph Type:     {model.get('graph_type', 'N/A')}")
    print(f"  Fixed n_blocks: {model.get('fixed_n_blocks', 'None')}")
    print(f"{'='*70}\n")
