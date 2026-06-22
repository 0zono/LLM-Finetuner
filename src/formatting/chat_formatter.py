from __future__ import annotations

from src.core.models import CanonicalRecord, TaskType


def format_records(
    records: list[CanonicalRecord], system_prompt: str
) -> tuple[list[CanonicalRecord], list[CanonicalRecord]]:
    formatted: list[CanonicalRecord] = []
    rejected: list[CanonicalRecord] = []

    for record in records:
        if record.task_type == TaskType.TOOL_CALLING:
            payload = _tool_calling(record, system_prompt)
        elif record.task_type == TaskType.INSTRUCTION_FOLLOWING:
            payload = _instruction(record)
        else:
            payload = _chat(record, system_prompt)

        if payload is None:
            rejected.append(record)
            continue
        record.payload = payload
        record.add_event("formatting", "formatted", {"task_type": record.task_type.value})
        formatted.append(record)
    return formatted, rejected


def _tool_calling(record: CanonicalRecord, system_prompt: str) -> dict | None:
    if not record.tool:
        if record.expected_output is None or not str(record.expected_output).strip():
            record.add_error(
                "formatting",
                "MISSING_TOOL_OR_RESPONSE",
                "Informe uma ferramenta ou uma resposta esperada sem ferramenta",
            )
            return None
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": record.content},
                {"role": "assistant", "content": str(record.expected_output)},
            ]
        }
    if not record.arguments:
        record.add_error("formatting", "MISSING_ARGUMENTS", "Argumentos ausentes")
        return None
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record.content},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"name": record.tool, "arguments": record.arguments}],
            },
        ]
    }


def _instruction(record: CanonicalRecord) -> dict | None:
    if record.expected_output is None or not str(record.expected_output).strip():
        record.add_error("formatting", "MISSING_OUTPUT", "Resposta esperada ausente")
        return None
    return {
        "instruction": record.content,
        "input": str(record.meta.get("input", "")),
        "output": str(record.expected_output),
    }


def _chat(record: CanonicalRecord, system_prompt: str) -> dict | None:
    if record.messages:
        return {"messages": record.messages}
    if record.expected_output is None or not str(record.expected_output).strip():
        record.add_error("formatting", "MISSING_OUTPUT", "Resposta esperada ausente")
        return None
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record.content},
            {"role": "assistant", "content": str(record.expected_output)},
        ]
    }


# Compatibilidade com o nome usado na primeira versão do protótipo.
def format_as_chat(records: list[CanonicalRecord]) -> list[CanonicalRecord]:
    formatted, _ = format_records(records, "Você é um assistente útil.")
    return formatted
