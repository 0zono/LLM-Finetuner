from __future__ import annotations

import argparse
import json
import sys

from pydantic import ValidationError

from src.core.config import PipelineConfig
from src.core.orchestrator import PipelineOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de datasets para fine-tuning")
    parser.add_argument("--config", default="config.yaml", help="Arquivo YAML de configuração")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = PipelineConfig.from_yaml(args.config)
        result = PipelineOrchestrator(config).run()
    except (OSError, ValueError, ValidationError, RuntimeError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1

    summary = {
        key: value
        for key, value in result.items()
        if key not in {"manifest", "evaluation"}
    }
    if result.get("evaluation"):
        summary["validator_f1"] = result["evaluation"]["f1_valid"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artefatos: {config.output_dir}")
    print(f"Manifesto: {config.reports_dir}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
