import pytest

from src.core.tool_registry import ToolArgumentsError, ToolRegistry


def test_registry_loads_domain_without_python_models() -> None:
    registry = ToolRegistry.from_file("domains/biblioteca/tools.json")
    assert registry.domain == "biblioteca"
    assert "reservar_livro" in registry.names
    assert len(registry.as_openai_tools()) == 5


def test_registry_validates_json_schema() -> None:
    registry = ToolRegistry.from_file("domains/biblioteca/tools.json")
    valid = registry.validate_arguments(
        "reservar_livro",
        {"usuario_id": 42, "isbn": "978-8535914849"},
    )
    assert valid["usuario_id"] == 42

    with pytest.raises(ToolArgumentsError):
        registry.validate_arguments(
            "reservar_livro",
            {"usuario_id": 42, "isbn": "inválido"},
        )
