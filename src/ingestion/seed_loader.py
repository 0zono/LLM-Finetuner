from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.core.models import CanonicalRecord, TaskType


def stable_id(source: str, index: int, item: dict[str, Any]) -> str:
    raw = json.dumps(
        {"source": source, "index": index, "item": item},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def load_seed_records(
    path: str | Path, task_type: TaskType = TaskType.TOOL_CALLING
) -> list[CanonicalRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de sementes não encontrado: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("O arquivo de sementes deve conter uma lista JSON")

    records: list[CanonicalRecord] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Semente {index} deve ser um objeto JSON")
        identifier = stable_id(str(path), index, item)
        content = item.get("text") or item.get("instruction") or item.get("content") or ""
        try:
            record = CanonicalRecord(
                id=identifier,
                source=str(path),
                source_id=f"{path.name}:{index}",
                parent_seed_id=identifier,
                task_type=task_type,
                content=content,
                expected_output=item.get("output", item.get("response")),
                tool=item.get("tool"),
                arguments=item.get("arguments", {}),
                messages=item.get("messages", []),
                meta={"original": item, "seed_index": index},
            )
        except ValidationError as error:
            raise ValueError(f"Semente inválida no índice {index}: {error}") from error
        record.add_event("ingestion", "loaded")
        records.append(record)
    return records
