from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.core.models import CanonicalRecord, RecordStatus, TaskType
from src.validation.schema_validator import validate_records


def evaluate_validator(
    valid_records: list[CanonicalRecord], report_path: str | Path
) -> dict[str, Any]:
    corpus: list[tuple[CanonicalRecord, bool, str]] = []
    for original in valid_records:
        baseline = deepcopy(original)
        baseline.status = RecordStatus.ACTIVE
        baseline.errors = []
        corpus.append((baseline, True, "valid"))
        if original.task_type == TaskType.TOOL_CALLING:
            corpus.extend(inject_tool_defects(original))

    predictions: list[dict[str, Any]] = []
    tp = fp = tn = fn = 0
    for record, expected_valid, category in corpus:
        record.status = RecordStatus.ACTIVE
        record.errors = []
        valid, invalid = validate_records([record])
        predicted_valid = bool(valid)
        if expected_valid and predicted_valid:
            tp += 1
        elif expected_valid and not predicted_valid:
            fn += 1
        elif not expected_valid and predicted_valid:
            fp += 1
        else:
            tn += 1
        predictions.append(
            {
                "id": record.id,
                "category": category,
                "expected_valid": expected_valid,
                "predicted_valid": predicted_valid,
                "errors": [error.code for error in record.errors],
            }
        )

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    report = {
        "examples": len(corpus),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision_valid": precision,
        "recall_valid": recall,
        "f1_valid": _safe_div(2 * precision * recall, precision + recall),
        "categories": dict(Counter(item["category"] for item in predictions)),
        "predictions": predictions,
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def inject_tool_defects(
    original: CanonicalRecord,
) -> list[tuple[CanonicalRecord, bool, str]]:
    mutations: list[tuple[CanonicalRecord, bool, str]] = []

    if original.tool is None:
        unexpected = deepcopy(original)
        unexpected.id += "-unexpected-tool"
        unexpected.payload["messages"][-1] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"name": "consultar_chamado", "arguments": {"chamado_id": 1}}
            ],
        }
        mutations.append((unexpected, False, "unexpected_tool_call"))
        return mutations

    unknown = deepcopy(original)
    unknown.id += "-unknown-tool"
    unknown.payload["messages"][-1]["tool_calls"][0]["name"] = "ferramenta_inexistente"
    mutations.append((unknown, False, "unknown_tool"))

    missing = deepcopy(original)
    missing.id += "-missing-argument"
    arguments = missing.payload["messages"][-1]["tool_calls"][0]["arguments"]
    if arguments:
        arguments.pop(next(iter(arguments)))
        mutations.append((missing, False, "missing_argument"))

    extra = deepcopy(original)
    extra.id += "-extra-argument"
    extra.payload["messages"][-1]["tool_calls"][0]["arguments"]["campo_extra"] = True
    mutations.append((extra, False, "extra_argument"))

    wrong_type = deepcopy(original)
    wrong_type.id += "-wrong-type"
    wrong_args = wrong_type.payload["messages"][-1]["tool_calls"][0]["arguments"]
    numeric_key = next((key for key, value in wrong_args.items() if isinstance(value, int)), None)
    if numeric_key:
        wrong_args[numeric_key] = "nao-e-numero"
        mutations.append((wrong_type, False, "wrong_type"))
    return mutations


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
