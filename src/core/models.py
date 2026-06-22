from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskType(StrEnum):
    TOOL_CALLING = "tool_calling"
    INSTRUCTION_FOLLOWING = "instruction_following"
    CHAT = "chat"


class RecordStatus(StrEnum):
    ACTIVE = "active"
    REJECTED = "rejected"
    VALID = "valid"


class ProcessingError(BaseModel):
    stage: str
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ProcessingEvent(BaseModel):
    stage: str
    action: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    details: dict[str, Any] = Field(default_factory=dict)


class CanonicalRecord(BaseModel):
    """Representação interna comum a todas as tarefas da pipeline."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    source_id: str
    parent_seed_id: str
    task_type: TaskType = TaskType.TOOL_CALLING
    content: str
    expected_output: Any | None = None
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    payload: dict[str, Any] | None = None
    status: RecordStatus = RecordStatus.ACTIVE
    errors: list[ProcessingError] = Field(default_factory=list)
    history: list[ProcessingEvent] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    def add_error(
        self,
        stage: str,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.errors.append(
            ProcessingError(
                stage=stage,
                code=code,
                message=message,
                details=details or {},
            )
        )
        self.status = RecordStatus.REJECTED

    def add_event(
        self, stage: str, action: str, details: dict[str, Any] | None = None
    ) -> None:
        self.history.append(
            ProcessingEvent(stage=stage, action=action, details=details or {})
        )


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

    @model_validator(mode="after")
    def validate_tool_call_role(self) -> "Message":
        if self.tool_calls and self.role != "assistant":
            raise ValueError("tool_calls só é permitido em mensagens assistant")
        if self.role in {"system", "user"} and not self.content:
            raise ValueError(f"mensagem {self.role} exige conteúdo")
        return self


class ChatExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[Message] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_sequence(self) -> "ChatExample":
        roles = [message.role for message in self.messages]
        if roles[0] == "assistant":
            raise ValueError("a conversa não pode iniciar com assistant")
        if "user" not in roles or "assistant" not in roles:
            raise ValueError("a conversa exige ao menos user e assistant")
        return self


class InstructionExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction: str = Field(min_length=1)
    input: str = ""
    output: str = Field(min_length=1)
