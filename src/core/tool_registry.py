from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserLookup(StrictModel):
    usuario_id: int | None = Field(default=None, gt=0)
    email: EmailStr | None = None

    @model_validator(mode="after")
    def exactly_one_identifier(self) -> "UserLookup":
        if (self.usuario_id is None) == (self.email is None):
            raise ValueError("informe exatamente um entre usuario_id e email")
        return self


class OpenTicket(StrictModel):
    usuario_id: int = Field(gt=0)
    categoria: Literal["login", "acesso", "hardware", "software", "rede", "outro"]
    descricao: str = Field(min_length=3, max_length=1000)
    prioridade: Literal["baixa", "media", "alta", "critica"]


class TicketLookup(StrictModel):
    chamado_id: int = Field(gt=0)


class ChangeTicketStatus(StrictModel):
    chamado_id: int = Field(gt=0)
    status: Literal["aberto", "em_andamento", "pendente", "resolvido", "fechado"]


TOOL_MODELS: dict[str, type[BaseModel]] = {
    "consultar_usuario": UserLookup,
    "abrir_chamado": OpenTicket,
    "consultar_chamado": TicketLookup,
    "alterar_status_chamado": ChangeTicketStatus,
    "listar_permissoes_usuario": UserLookup,
}


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in TOOL_MODELS:
        raise KeyError(name)
    return TOOL_MODELS[name].model_validate(arguments).model_dump(exclude_none=True)


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Ferramenta de suporte técnico: {name}",
                "parameters": model.model_json_schema(),
            },
        }
        for name, model in TOOL_MODELS.items()
    ]
