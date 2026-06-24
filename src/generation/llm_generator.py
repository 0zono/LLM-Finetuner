from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from src.core.llm_client import LocalLLMClient
from src.core.models import CanonicalRecord
from src.preprocessing.cleaner import normalize_text


GENERATION_PROMPT = (
    "Você gera paráfrases de uma mensagem do USUÁRIO para um dataset de fine-tuning. "
    "Nunca responda à mensagem e nunca escreva como assistente. Cada variação deve ser "
    "algo que um usuário enviaria antes da resposta. Preserve rigorosamente a intenção, "
    "os identificadores, números, e-mails e entidades. Não adicione fatos ausentes, não "
    "copie a mensagem original, não mencione nomes internos de ferramentas ou funções que "
    "não apareçam no original e produza textos únicos. Responda somente com JSON no "
    "formato {\"variations\": [\"...\"]}."
)


def generate_examples(
    records: list[CanonicalRecord],
    *,
    enabled: bool,
    variations_per_seed: int,
    client: LocalLLMClient | None = None,
) -> tuple[list[CanonicalRecord], list[CanonicalRecord]]:
    if not enabled or variations_per_seed == 0:
        return list(records), []
    if client is None:
        raise ValueError("Geração habilitada exige um cliente de LM")

    output = list(records)
    rejected: list[CanonicalRecord] = []
    requested_count = variations_per_seed + max(2, variations_per_seed // 4)

    for record in records:
        try:
            response = client.chat_json(
                [
                    {"role": "system", "content": GENERATION_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "target_role": "user",
                                "task_type": record.task_type.value,
                                "original_user_message": record.content,
                                "tool": record.tool,
                                "arguments": record.arguments,
                                "count": requested_count,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                schema={
                    "type": "object",
                    "properties": {
                        "variations": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "minItems": variations_per_seed,
                            "maxItems": requested_count,
                        }
                    },
                    "required": ["variations"],
                    "additionalProperties": False,
                },
            )
            candidates = response.get("variations", [])
            accepted = _accept_candidates(
                record,
                candidates,
                variations_per_seed,
                output,
                rejected,
            )
            if accepted < variations_per_seed:
                failed = deepcopy(record)
                failed.id = f"{record.id}-insufficient-variations"
                failed.add_error(
                    "generation",
                    "INSUFFICIENT_UNIQUE_VARIATIONS",
                    f"Esperadas {variations_per_seed}, aceitas {accepted}",
                    {"expected": variations_per_seed, "accepted": accepted},
                )
                rejected.append(failed)
        except Exception as error:
            failed = deepcopy(record)
            failed.id = f"{record.id}-generation-error"
            failed.add_error("generation", "LLM_GENERATION_FAILED", str(error))
            rejected.append(failed)
    return output, rejected


def _accept_candidates(
    record: CanonicalRecord,
    candidates: list[Any],
    limit: int,
    output: list[CanonicalRecord],
    rejected: list[CanonicalRecord],
) -> int:
    original_key = normalize_text(record.content).casefold()
    seen = {original_key}
    required_literals = _stable_literals(record.arguments)
    accepted = 0

    for candidate_index, candidate in enumerate(candidates, start=1):
        if accepted >= limit:
            break
        text = normalize_text(candidate) if isinstance(candidate, str) else ""
        key = text.casefold()
        error_code = None
        error_message = None
        if not text:
            error_code, error_message = "EMPTY_GENERATION", "Variação vazia"
        elif key in seen:
            error_code, error_message = "DUPLICATE_GENERATION", "Variação duplicada ou igual à semente"
        elif not _contains_literals(text, required_literals):
            error_code, error_message = "ENTITY_NOT_PRESERVED", "Identificador, número ou e-mail não foi preservado"

        if error_code:
            rejected.append(
                _rejected_candidate(record, candidate_index, text, error_code, error_message)
            )
            continue

        seen.add(key)
        accepted += 1
        variation = deepcopy(record)
        digest = hashlib.sha256(
            f"{record.parent_seed_id}:{accepted}:{text}".encode("utf-8")
        ).hexdigest()[:20]
        variation.id = digest
        variation.source = "local_llm"
        variation.source_id = f"{record.source_id}:generated:{accepted}"
        variation.content = text
        variation.payload = None
        variation.history = list(record.history)
        variation.meta["generated"] = True
        variation.meta["source_content"] = record.content
        variation.meta["candidate_index"] = candidate_index
        variation.add_event(
            "generation",
            "generated",
            {"parent_id": record.id, "index": accepted, "candidate_index": candidate_index},
        )
        output.append(variation)
    return accepted


def _rejected_candidate(
    record: CanonicalRecord,
    index: int,
    text: str,
    code: str,
    message: str,
) -> CanonicalRecord:
    rejected = deepcopy(record)
    rejected.id = f"{record.id}-candidate-{index}-rejected"
    rejected.source = "local_llm"
    rejected.source_id = f"{record.source_id}:rejected:{index}"
    rejected.content = text
    rejected.meta["generated"] = True
    rejected.meta["source_content"] = record.content
    rejected.add_error("generation", code, message, {"candidate_index": index})
    return rejected


def _stable_literals(arguments: dict[str, Any]) -> set[str]:
    literals: set[str] = set()
    for value in arguments.values():
        if isinstance(value, int) and not isinstance(value, bool):
            literals.add(str(value))
        elif isinstance(value, str) and ("@" in value or re.fullmatch(r"[0-9-]{6,}", value)):
            literals.add(value.casefold())
    return literals


def _contains_literals(text: str, literals: set[str]) -> bool:
    normalized = text.casefold()
    return all(literal in normalized for literal in literals)
