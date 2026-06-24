import json

from src.core.models import CanonicalRecord
from src.curation.llm_judge import heuristic_score
from src.generation.llm_generator import generate_examples


class FakeGenerationClient:
    def __init__(self, variations):
        self.variations = variations
        self.requests = []

    def chat_json(self, messages, **kwargs):
        self.requests.append(messages)
        return {"variations": self.variations}


def record(content="Obrigado pela ajuda!", expected_output="Disponha!"):
    return CanonicalRecord(
        id="seed-1",
        source="test",
        source_id="test:1",
        parent_seed_id="seed-1",
        content=content,
        expected_output=expected_output,
        meta={"split": "train"},
    )


def test_generator_hides_expected_output_and_rejects_duplicates() -> None:
    client = FakeGenerationClient(
        ["Obrigado pela ajuda!", "Muito obrigado pelo atendimento!", "Agradeço pela ajuda!"]
    )
    output, rejected = generate_examples(
        [record()], enabled=True, variations_per_seed=2, client=client
    )
    assert len(output) == 3
    assert any(error.code == "DUPLICATE_GENERATION" for item in rejected for error in item.errors)
    request_data = json.loads(client.requests[0][1]["content"])
    assert "expected_output" not in request_data
    assert request_data["target_role"] == "user"


def test_heuristic_rejects_role_inversion() -> None:
    gratitude_reply = record(content="Disponha! Estou à disposição.")
    gratitude_reply.meta.update(
        generated=True,
        source_content="Obrigado pela ajuda!",
    )
    gratitude_reply.payload = {"messages": []}
    score, _ = heuristic_score(gratitude_reply)
    assert score < 0.7

    answer_instead_of_request = record(
        content="O suporte atende problemas de login e rede.",
        expected_output="Resposta esperada",
    )
    answer_instead_of_request.meta.update(
        generated=True,
        source_content="Explique quais problemas o suporte atende.",
    )
    answer_instead_of_request.payload = {"messages": []}
    score, _ = heuristic_score(answer_instead_of_request)
    assert score < 0.7
