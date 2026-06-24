from copy import deepcopy

from src.core.models import CanonicalRecord, TaskType
from src.core.tool_registry import ToolRegistry
from src.formatting.chat_formatter import format_records
from src.validation.schema_validator import validate_records


REGISTRY = ToolRegistry.from_file("domains/suporte_tecnico/tools.json")


def tool_record(**overrides) -> CanonicalRecord:
    data = {
        "id": "seed-1",
        "source": "test",
        "source_id": "test:1",
        "parent_seed_id": "seed-1",
        "task_type": TaskType.TOOL_CALLING,
        "content": "Consulte os dados do usuário joao@example.com.",
        "tool": "consultar_usuario",
        "arguments": {"email": "joao@example.com"},
    }
    data.update(overrides)
    return CanonicalRecord(**data)


def test_valid_tool_call() -> None:
    formatted, rejected = format_records([tool_record()], "Sistema")
    valid, invalid = validate_records(formatted, REGISTRY)
    assert len(valid) == 1
    assert rejected == []
    assert invalid == []


def test_unknown_tool_is_rejected() -> None:
    record = tool_record(tool="nao_existe")
    formatted, _ = format_records([record], "Sistema")
    valid, invalid = validate_records(formatted, REGISTRY)
    assert valid == []
    assert invalid[0].errors[-1].code == "UNKNOWN_TOOL"


def test_extra_argument_is_rejected() -> None:
    record = tool_record(arguments={"email": "joao@example.com", "extra": True})
    formatted, _ = format_records([record], "Sistema")
    valid, invalid = validate_records(formatted, REGISTRY)
    assert valid == []
    assert invalid[0].errors[-1].code == "EXTRA_FIELD"


def test_exactly_one_user_identifier() -> None:
    record = tool_record(arguments={"email": "joao@example.com", "usuario_id": 1})
    formatted, _ = format_records([record], "Sistema")
    valid, invalid = validate_records(formatted, REGISTRY)
    assert valid == []
    assert invalid


def test_instruction_following() -> None:
    record = CanonicalRecord(
        id="i1",
        source="test",
        source_id="test:1",
        parent_seed_id="i1",
        task_type=TaskType.INSTRUCTION_FOLLOWING,
        content="Resuma o texto.",
        expected_output="Resumo esperado.",
    )
    formatted, rejected = format_records([record], "Sistema")
    valid, invalid = validate_records(formatted, REGISTRY)
    assert rejected == []
    assert len(valid) == 1
    assert invalid == []


def test_chat_without_output_is_tracked() -> None:
    record = CanonicalRecord(
        id="c1",
        source="test",
        source_id="test:1",
        parent_seed_id="c1",
        task_type=TaskType.CHAT,
        content="Olá",
    )
    formatted, rejected = format_records([record], "Sistema")
    assert formatted == []
    assert rejected[0].errors[-1].code == "MISSING_OUTPUT"


def test_tool_calling_allows_intent_without_tool() -> None:
    record = tool_record(
        tool=None,
        arguments={},
        content="Qual é o horário do suporte?",
        expected_output="O suporte funciona em horário comercial.",
    )
    formatted, rejected = format_records([record], "Sistema")
    valid, invalid = validate_records(formatted, REGISTRY)
    assert rejected == []
    assert len(valid) == 1
    assert invalid == []
