import re
import unicodedata

from src.core.models import CanonicalRecord


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def clean_records(
    records: list[CanonicalRecord], min_length: int = 3, max_length: int = 10_000
) -> tuple[list[CanonicalRecord], list[CanonicalRecord]]:
    cleaned: list[CanonicalRecord] = []
    rejected: list[CanonicalRecord] = []
    seen_contents: set[str] = set()

    for record in records:
        content = normalize_text(record.content)
        key = content.casefold()
        if not content:
            record.add_error("preprocessing", "EMPTY_CONTENT", "Conteúdo vazio")
        elif len(content) < min_length:
            record.add_error("preprocessing", "CONTENT_TOO_SHORT", "Conteúdo muito curto")
        elif len(content) > max_length:
            record.add_error("preprocessing", "CONTENT_TOO_LONG", "Conteúdo muito longo")
        elif key in seen_contents:
            record.add_error("preprocessing", "EXACT_DUPLICATE", "Conteúdo duplicado")

        if record.errors:
            rejected.append(record)
            continue

        seen_contents.add(key)
        record.content = content
        record.add_event("preprocessing", "normalized")
        cleaned.append(record)

    return cleaned, rejected
