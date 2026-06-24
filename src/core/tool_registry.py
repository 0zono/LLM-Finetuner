from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any]


class ToolRegistryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(min_length=1)
    version: str = "1.0"
    tools: list[ToolDefinition] = Field(min_length=1)


class ToolArgumentsError(ValueError):
    def __init__(self, tool_name: str, errors: list[dict[str, Any]]) -> None:
        self.tool_name = tool_name
        self.errors = errors
        message = "; ".join(error["message"] for error in errors)
        super().__init__(f"Argumentos inválidos para {tool_name}: {message}")


class ToolRegistry:
    """Registro de ferramentas independente de domínio, baseado em JSON Schema."""

    def __init__(self, document: ToolRegistryDocument) -> None:
        self.domain = document.domain
        self.version = document.version
        self._tools = {tool.name: tool for tool in document.tools}
        if len(self._tools) != len(document.tools):
            raise ValueError("O registro contém nomes de ferramentas duplicados")
        for tool in document.tools:
            if tool.parameters.get("type") != "object":
                raise ValueError(
                    f"O schema de parâmetros de {tool.name} deve ter type=object"
                )

    @classmethod
    def from_file(cls, path: str | Path) -> "ToolRegistry":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Registro de ferramentas não encontrado: {path}")
        with path.open("r", encoding="utf-8") as stream:
            document = ToolRegistryDocument.model_validate(json.load(stream))
        return cls(document)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def validate_arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(name)
        errors = _validate_schema(arguments, self._tools[name].parameters)
        if errors:
            raise ToolArgumentsError(name, errors)
        return arguments

    def as_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]


def _validate_schema(
    value: Any,
    schema: dict[str, Any],
    path: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Valida o subconjunto de JSON Schema usado pelos contratos do projeto."""
    path = path or []
    errors: list[dict[str, Any]] = []

    if "oneOf" in schema:
        matches = sum(not _validate_schema(value, option, path) for option in schema["oneOf"])
        if matches != 1:
            errors.append(_schema_error("oneOf", "deve satisfazer exatamente uma alternativa", path))

    expected_type = schema.get("type")
    type_valid = {
        "object": isinstance(value, dict),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
    }.get(expected_type, True)
    if not type_valid:
        return errors + [_schema_error("type", f"tipo esperado: {expected_type}", path)]

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                errors.append(_schema_error("required", f"campo obrigatório ausente: {required}", path + [required]))
        if schema.get("additionalProperties") is False:
            for key in value.keys() - properties.keys():
                errors.append(_schema_error("additionalProperties", f"campo não permitido: {key}", path + [key]))
        for key, item in value.items():
            if key in properties:
                errors.extend(_validate_schema(item, properties[key], path + [key]))

    if "enum" in schema and value not in schema["enum"]:
        errors.append(_schema_error("enum", f"valor não permitido: {value}", path))
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(_schema_error("minLength", "texto menor que o mínimo", path))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(_schema_error("maxLength", "texto maior que o máximo", path))
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            errors.append(_schema_error("pattern", "texto não corresponde ao padrão", path))
        if schema.get("format") == "email" and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            errors.append(_schema_error("format", "e-mail inválido", path))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(_schema_error("minimum", "valor abaixo do mínimo", path))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(_schema_error("maximum", "valor acima do máximo", path))
    return errors


def _schema_error(validator: str, message: str, path: list[str]) -> dict[str, Any]:
    return {"validator": validator, "message": message, "path": path}
