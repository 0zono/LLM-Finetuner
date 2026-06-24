from __future__ import annotations

import json
import re

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
        checks: dict[str, object] = {"heuristic_reason": reason}

        # Sementes humanas não precisam ser julgadas novamente pelo LM.
        if mode == "llm" and record.meta.get("generated") and score >= minimum_score:
            if client is None:
                raise ValueError("Curadoria LLM habilitada exige cliente de LM")
            try:
                result = semantic_judge(record, client)
                checks.update(result)
                semantic_ok = all(
                    [
                        bool(result["is_user_message"]),
                        bool(result["same_intent"]),
                        bool(result["entities_preserved"]),
                        not bool(result["adds_information"]),
                    ]
                )
                score = float(result["score"])
                if not semantic_ok:
                    score = min(score, 0.4)
                reason = str(result["reason"])
            except Exception as error:
                record.add_error("curation", "LLM_JUDGE_FAILED", str(error))
                rejected.append(record)
                continue

        record.meta["curation"] = {
            "mode": mode,
            "score": score,
            "reason": reason,
            "checks": checks,
        }
        record.add_event("curation", "scored", {"score": score, "mode": mode})
        if score >= minimum_score:
            approved.append(record)
        else:
            record.add_error(
                "curation",
                "LOW_QUALITY_SCORE",
                reason,
                {"score": score, "minimum": minimum_score, "checks": checks},
            )
            rejected.append(record)
    return approved, rejected


def semantic_judge(record: CanonicalRecord, client: LocalLLMClient) -> dict[str, object]:
    return client.chat_json(
        [
            {
                "role": "system",
                "content": (
                    "Compare uma semente humana e uma paráfrase candidata destinada ao papel USER. "
                    "A candidata não pode responder à semente, trocar de papel, mudar a intenção, "
                    "remover identificadores ou acrescentar fatos. Avalie estritamente e responda JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "original_user_message": record.meta.get("source_content"),
                        "candidate_user_message": record.content,
                        "tool": record.tool,
                        "arguments": record.arguments,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        schema={
            "type": "object",
            "properties": {
                "is_user_message": {"type": "boolean"},
                "same_intent": {"type": "boolean"},
                "entities_preserved": {"type": "boolean"},
                "adds_information": {"type": "boolean"},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": [
                "is_user_message",
                "same_intent",
                "entities_preserved",
                "adds_information",
                "score",
                "reason",
            ],
            "additionalProperties": False,
        },
        temperature=0,
    )


def heuristic_score(record: CanonicalRecord) -> tuple[float, str]:
    content = record.content.strip()
    if not content:
        return 0.0, "Conteúdo vazio"
    if len(content) < 10:
        return 0.3, "Conteúdo muito curto"
    if record.payload is None:
        return 0.0, "Payload ausente"

    if record.meta.get("generated"):
        original = str(record.meta.get("source_content", "")).strip()
        if record.tool and record.tool.casefold() in content.casefold() and record.tool.casefold() not in original.casefold():
            return 0.2, "A variação expõe o nome interno da ferramenta"
        if original and _is_request_like(original) and not _is_request_like(content):
            return 0.2, "A semente é uma solicitação, mas a variação parece uma resposta"
        if _contains_gratitude(original) and not _contains_gratitude(content):
            return 0.2, "A variação não preserva a intenção de agradecimento"

    return 1.0, "Aprovado pelos critérios heurísticos"


def _is_request_like(text: str) -> bool:
    normalized = text.strip().casefold()
    if "?" in normalized:
        return True
    first_words = [
        re.sub(r"[^a-záàâãéêíóôõúç]", "", word)
        for word in normalized.split()[:3]
    ]
    request_prefixes = (
        "abr", "acess", "alter", "atual", "bus", "colo", "consult", "cri",
        "defin", "demonstr", "exib", "expliqu", "ger", "gostari", "inform",
        "inici", "list", "localiz", "modif", "mostr", "mud", "obtenh", "poderi",
        "precis", "procur", "realiz", "verif", "quer",
    )
    interrogatives = ("qual", "quais", "como", "quando", "onde")
    return any(
        word.startswith(request_prefixes) or word in interrogatives
        for word in first_words
    ) or any(
        re.search(rf"\b{re.escape(marker.strip())}\b", normalized)
        for marker in ("quero", "preciso", "poderia", "por favor", "em que")
    )


def _contains_gratitude(text: str) -> bool:
    normalized = text.casefold()
    return any(marker in normalized for marker in ("obrigad", "agrade", "valeu", "grato", "grata"))
