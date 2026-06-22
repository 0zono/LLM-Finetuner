from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from src.core.llm_client import LocalLLMClient
from src.core.models import CanonicalRecord, TaskType


def generate_examples(
    records: list[CanonicalRecord],
    *,
    enabled: bool,
    variations_per_seed: int,
    client: LocalLLMClient | None = None,
) -> tuple[list[CanonicalRecord], list[CanonicalRecord]]:
    if not enabled or variations_per_seed == 0:
        return records, []
    if client is None:
        raise ValueError("Geração habilitada exige um cliente de LM")

    output = list(records)
    rejected: list[CanonicalRecord] = []
    for record in records:
        try:
            response = client.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "Gere paráfrases diversas para um dataset de fine-tuning. "
                            "Preserve rigorosamente a intenção, entidades, valores e resposta esperada. "
                            "Responda apenas JSON no formato {\"variations\": [\"...\"]}."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "task_type": record.task_type.value,
                                "text": record.content,
                                "tool": record.tool,
                                "arguments": record.arguments,
                                "expected_output": record.expected_output,
                                "count": variations_per_seed,
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
                            "items": {"type": "string"},
                            "minItems": variations_per_seed,
                            "maxItems": variations_per_seed,
                        }
                    },
                    "required": ["variations"],
                    "additionalProperties": False,
                },
            )
            variations = response.get("variations", [])[:variations_per_seed]
            if len(variations) != variations_per_seed:
                raise ValueError("LM retornou quantidade incorreta de variações")
            for index, text in enumerate(variations, start=1):
                variation = deepcopy(record)
                digest = hashlib.sha256(
                    f"{record.parent_seed_id}:{index}:{text}".encode("utf-8")
                ).hexdigest()[:20]
                variation.id = digest
                variation.source = "local_llm"
                variation.source_id = f"{record.source_id}:generated:{index}"
                variation.content = text.strip()
                variation.payload = None
                variation.history = list(record.history)
                variation.add_event(
                    "generation", "generated", {"parent_id": record.id, "index": index}
                )
                variation.meta["generated"] = True
                output.append(variation)
        except Exception as error:
            failed = deepcopy(record)
            failed.id = f"{record.id}-generation-error"
            failed.add_error("generation", "LLM_GENERATION_FAILED", str(error))
            rejected.append(failed)
    return output, rejected
