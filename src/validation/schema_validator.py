from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from src.core.models import (
    CanonicalRecord,
    ChatExample,
    InstructionExample,
    RecordStatus,
    TaskType,
)
from src.core.tool_registry import ToolArgumentsError, ToolRegistry


def validate_records(
    records: list[CanonicalRecord],
    tool_registry: ToolRegistry | None = None,
) -> tuple[list[CanonicalRecord], list[CanonicalRecord]]:
    valid: list[CanonicalRecord] = []
    invalid: list[CanonicalRecord] = []

    for record in records:
        if record.payload is None:
            record.add_error("validation", "MISSING_PAYLOAD", "Payload ausente")
            invalid.append(record)
            continue
        try:
            json.dumps(record.payload, ensure_ascii=False, allow_nan=False)
            if record.task_type == TaskType.INSTRUCTION_FOLLOWING:
                InstructionExample.model_validate(record.payload)
            else:
                example = ChatExample.model_validate(record.payload)
                if record.task_type == TaskType.TOOL_CALLING:
                    if tool_registry is None:
                        raise ValueError("validação de tool calling exige um registro de ferramentas")
                    _validate_tool_calling(record, example, tool_registry)
            record.status = RecordStatus.VALID
            record.meta["valid"] = True
            record.add_event("validation", "approved")
            valid.append(record)
        except KeyError as error:
            record.add_error(
                "validation",
                "UNKNOWN_TOOL",
                f"Ferramenta desconhecida: {error.args[0]}",
            )
            invalid.append(record)
        except ToolArgumentsError as error:
            code = classify_tool_error(error)
            record.add_error(
                "validation",
                code,
                str(error),
                {"tool": error.tool_name, "schema_errors": error.errors},
            )
            invalid.append(record)
        except (ValidationError, ValueError, TypeError) as error:
            code = classify_validation_error(error)
            details = (
                {"errors": sanitize_errors(error.errors())}
                if isinstance(error, ValidationError)
                else {}
            )
            record.add_error("validation", code, str(error), details)
            invalid.append(record)
    return valid, invalid


def _validate_tool_calling(
    record: CanonicalRecord,
    example: ChatExample,
    tool_registry: ToolRegistry,
) -> None:
    calls = [
        call
        for message in example.messages
        for call in (message.tool_calls or [])
    ]
    if record.tool is None:
        if calls:
            raise ValueError("exemplo anotado sem ferramenta contém tool_call")
        assistant_messages = [m for m in example.messages if m.role == "assistant"]
        if not assistant_messages or not assistant_messages[-1].content:
            raise ValueError("exemplo sem ferramenta exige resposta textual")
        return
    if len(calls) != 1:
        raise ValueError("o exemplo deve conter exatamente uma chamada de ferramenta")
    call = calls[0]
    validated = tool_registry.validate_arguments(call.name, call.arguments)
    if record.tool and call.name != record.tool:
        raise ValueError("ferramenta no payload diverge da anotação canônica")
    if record.arguments and validated != record.arguments:
        raise ValueError("argumentos no payload divergem da anotação canônica")


def classify_tool_error(error: ToolArgumentsError) -> str:
    validators = {item.get("validator") for item in error.errors}
    if "additionalProperties" in validators:
        return "EXTRA_FIELD"
    if "required" in validators:
        return "MISSING_FIELD"
    if "type" in validators:
        return "INVALID_TYPE"
    if "enum" in validators:
        return "INVALID_ENUM"
    if "oneOf" in validators or "anyOf" in validators:
        return "INVALID_ALTERNATIVE"
    if "format" in validators or "pattern" in validators:
        return "INVALID_FORMAT"
    return "INVALID_ARGUMENT_VALUE"


def classify_validation_error(error: Exception) -> str:
    text = str(error).casefold()
    if "extra_forbidden" in text or "extra inputs" in text:
        return "EXTRA_FIELD"
    if "missing" in text or "field required" in text:
        return "MISSING_FIELD"
    if "literal_error" in text or "input should be" in text:
        return "INVALID_ENUM"
    if "int_" in text or "string_" in text or "type" in text:
        return "INVALID_TYPE"
    if "exatamente uma chamada" in text:
        return "INVALID_TOOL_CALL_COUNT"
    if "sem ferramenta" in text:
        return "UNEXPECTED_TOOL_CALL"
    if "diverge" in text:
        return "ANNOTATION_MISMATCH"
    return "SCHEMA_VALIDATION_ERROR"


def sanitize_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for item in errors:
        clean.append(
            {
                "type": item.get("type"),
                "loc": list(item.get("loc", [])),
                "msg": item.get("msg"),
                "input": repr(item.get("input")),
            }
        )
    return clean
