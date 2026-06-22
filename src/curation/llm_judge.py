from __future__ import annotations

import json
from copy import deepcopy

from src.core.llm_client import LocalLLMClient
from src.core.models import CanonicalRecord


def curate_records(
    records: list[CanonicalRecord],
    *,
    enabled: bool,
    mode: str,
    minimum_score: float,
    client: LocalLLMClient | None = None,
) -> tuple[list[CanonicalRecord], list[CanonicalRecord]]:
    if not enabled:
        return records, []

    approved: list[CanonicalRecord] = []
    rejected: list[CanonicalRecord] = []
    for record in records:
        score, reason = heuristic_score(record)
        if mode == "llm" and score >= minimum_score:
            if client is None:
                raise ValueError("Curadoria LLM habilitada exige cliente de LM")
            try:
                result = client.chat_json(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Avalie um exemplo de fine-tuning quanto a clareza, coerência, "
                                "correção e utilidade. Responda JSON com score entre 0 e 1 e reason."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(record.payload, ensure_ascii=False),
                        },
                    ],
                    schema={
                        "type": "object",
                        "properties": {
                            "score": {"type": "number", "minimum": 0, "maximum": 1},
                            "reason": {"type": "string"},
                        },
                        "required": ["score", "reason"],
                        "additionalProperties": False,
                    },
                    temperature=0,
                )
                score = float(result["score"])
                reason = str(result["reason"])
            except Exception as error:
                record.add_error("curation", "LLM_JUDGE_FAILED", str(error))
                rejected.append(record)
                continue

        record.meta["curation"] = {"mode": mode, "score": score, "reason": reason}
        record.add_event("curation", "scored", {"score": score, "mode": mode})
        if score >= minimum_score:
            approved.append(record)
        else:
            record.add_error(
                "curation",
                "LOW_QUALITY_SCORE",
                reason,
                {"score": score, "minimum": minimum_score},
            )
            rejected.append(record)
    return approved, rejected


def heuristic_score(record: CanonicalRecord) -> tuple[float, str]:
    if not record.content.strip():
        return 0.0, "Conteúdo vazio"
    if len(record.content.strip()) < 10:
        return 0.3, "Conteúdo muito curto"
    if record.payload is None:
        return 0.0, "Payload ausente"
    return 1.0, "Aprovado pelos critérios heurísticos"
