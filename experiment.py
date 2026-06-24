from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.core.config import PipelineConfig
from src.core.orchestrator import PipelineOrchestrator


ABLATIONS = {
    "baseline": {"enable_generation": False, "enable_curation": False},
    "generation": {"enable_generation": True, "enable_curation": False},
    "heuristic": {
        "enable_generation": True,
        "enable_curation": True,
        "curation_mode": "heuristic",
    },
    "llm_judge": {
        "enable_generation": True,
        "enable_curation": True,
        "curation_mode": "llm",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa estudo de ablação")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="reports/ablation.csv")
    args = parser.parse_args()
    base = PipelineConfig.from_yaml(args.config)
    rows: list[dict[str, object]] = []

    for name, overrides in ABLATIONS.items():
        config = base.model_copy(update=overrides, deep=True)
        config.output_dir = f"data/output/ablation/{base.domain}/{name}"
        config.reports_dir = f"reports/ablation/{base.domain}/{name}"
        result = PipelineOrchestrator(config).run()
        evaluation = result.get("evaluation") or {}
        rows.append(
            {
                "configuration": name,
                "total": result["total_records"],
                "generated": result["generated_records"],
                "valid": result["valid_records"],
                "invalid": result["invalid_records"],
                "validator_f1": evaluation.get("f1_valid", ""),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Relatório de ablação: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
