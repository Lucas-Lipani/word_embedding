import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class ConfigManager:
    """
    Gerencia pastas de configuração (config_NNN) dentro de uma seed.
    Cada pasta agrupa runs com a mesma configuração (sbm_mode, graph_type, n_blocks).
    """

    def __init__(self, seed_dir: Path):
        """
        :param seed_dir: Caminho da pasta seed (ex: outputs/partitions/100/seed_42)
        """
        self.seed_dir = Path(seed_dir)
        self.seed_dir.mkdir(parents=True, exist_ok=True)

    def _get_config_signature(
        self,
        nested: bool,
        graph_type: str,
        n_blocks: int | None,
    ) -> str:
        """
        Cria uma assinatura única para uma configuração.
        Compara: sbm_mode, graph_type, fixed_n_blocks
        """
        return json.dumps(
            {
                "sbm_mode": "nested" if nested else "flat",
                "graph_type": graph_type,
                "fixed_n_blocks": n_blocks,
            },
            sort_keys=True,
        )

    def find_or_create_config_dir(
        self,
        nested: bool,
        graph_type: str,
        n_blocks: int | None,
    ) -> tuple[Path, int, bool]:
        """
        Procura uma pasta config_NNN que já tenha a mesma assinatura.
        Se encontrar, retorna seu caminho e índice (reutiliza=True).
        Se não encontrar, cria uma nova com índice sequencial (reutiliza=False).

        :return: (caminho da pasta config_NNN, índice, foi_reutilizada)
        """
        target_sig = self._get_config_signature(nested, graph_type, n_blocks)

        # Procura pastas config_* existentes
        config_dirs = sorted(self.seed_dir.glob("config_*"))

        for config_dir in config_dirs:
            config_file = config_dir / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        saved_cfg = json.load(f)

                    saved_sig = json.dumps(
                        {
                            "sbm_mode": saved_cfg["sbm_mode"],
                            "graph_type": saved_cfg["graph_type"],
                            "fixed_n_blocks": saved_cfg["fixed_n_blocks"],
                        },
                        sort_keys=True,
                    )

                    if saved_sig == target_sig:
                        idx = int(config_dir.name.split("_")[1])
                        print(
                            f"[CONFIG] Reutilizando pasta existente: {config_dir.name}"
                        )
                        return config_dir, idx, True
                except Exception as e:
                    print(f"[WARN] Erro ao ler {config_file}: {e}")

        # Se não encontrou, cria nova pasta com índice sequencial
        next_idx = len(config_dirs) + 1
        new_config_dir = self.seed_dir / f"config_{next_idx:03d}"
        new_config_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[CONFIG] Criando nova pasta de configuração: {new_config_dir.name}"
        )

        return new_config_dir, next_idx, False

    def save_config(
        self,
        config_dir: Path,
        n_samples: int,
        seed: int,
        nested: bool,
        n_blocks: int | None,
        graph_type: str,
    ) -> Path:
        """
        Salva config.json na pasta config_dir (apenas se não existir).

        :return: Caminho do config.json salvo
        """
        config_file = config_dir / "config.json"

        # Se já existe, não sobrescreve
        if config_file.exists():
            print(
                f"[CONFIG] config.json já existe em {config_dir.name}, pulando..."
            )
            return config_file

        cfg_data = {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "n_samples": n_samples,
            "sbm_mode": "nested" if nested else "flat",
            "graph_type": graph_type,
            "fixed_n_blocks": n_blocks,
        }

        with open(config_file, "w") as f:
            json.dump(cfg_data, f, indent=2)

        print(f"[CONFIG] Metadados salvos em: {config_file}")
        return config_file


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
    print(f"Seed:             {cfg.get('seed', 'N/A')}")
    print(f"Samples:          {cfg.get('n_samples', 'N/A')}")
    print(f"SBM Mode:         {cfg.get('sbm_mode', 'N/A')}")
    print(f"Graph Type:       {cfg.get('graph_type', 'N/A')}")
    print(f"Fixed n_blocks:   {cfg.get('fixed_n_blocks', 'None')}")
    print(f"{'='*70}\n")
