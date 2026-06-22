from src.core.models import CanonicalRecord
from src.preprocessing.cleaner import clean_records, normalize_text


def record(identifier: str, content: str) -> CanonicalRecord:
    return CanonicalRecord(
        id=identifier,
        source="test",
        source_id=identifier,
        parent_seed_id=identifier,
        content=content,
    )


def test_normalization_and_duplicate_tracking() -> None:
    cleaned, rejected = clean_records(
        [record("1", "  Texto   válido "), record("2", "texto válido")]
    )
    assert cleaned[0].content == "Texto válido"
    assert rejected[0].errors[-1].code == "EXACT_DUPLICATE"


def test_unicode_normalization() -> None:
    assert normalize_text("Cafe\u0301") == "Café"
