"""Tests for PDF generation using the Streamlit API response shape."""

from io import BytesIO

import pytest
from pypdf import PdfReader
from utils.pdf import generate_plan_pdf


@pytest.fixture
def sample_plan():
    return {
        "targets": {
            "calories": 2100,
            "protein_g": 130,
            "carbs_g": 240,
            "fat_g": 65,
        },
        "days": {
            "Day 1": {
                "meals": {
                    "breakfast": {
                        "title": "Overnight Oats",
                        "url": "https://example.com/oats",
                        "calories": 350,
                    },
                    "lunch": {
                        "title": "Chicken Salad",
                        "url": "https://example.com/salad",
                        "calories": 520,
                    },
                }
            },
            "Day 2": {
                "meals": {
                    "dinner": {
                        "title": "Salmon & Rice",
                        "url": "https://example.com/salmon",
                        "calories": 680,
                    },
                }
            },
        },
        "openai_api_key": "SECRET_SHOULD_NOT_APPEAR",
        "messages": ["PRIVATE_CHAT_SHOULD_NOT_APPEAR"],
    }


def _read(pdf_bytes: bytes) -> PdfReader:
    """Read a PDF from memory."""
    return PdfReader(BytesIO(pdf_bytes))


def test_generates_readable_pdf(sample_plan):
    data = generate_plan_pdf(sample_plan)

    assert isinstance(data, bytes)
    assert data.startswith(b"%PDF-")
    assert len(_read(data).pages) >= 1


def test_plan_content_and_targets_are_in_pdf(sample_plan):
    pdf = generate_plan_pdf(sample_plan)
    text = "\n".join(
        page.extract_text() for page in _read(pdf).pages
    )

    for expected in [
        "Day 1",
        "Day 2",
        "Overnight Oats",
        "Chicken Salad",
        "Salmon",
        "350 kcal",
        "2100 kcal",
        "130 g",
    ]:
        assert expected in text


def test_recipe_links_are_clickable(sample_plan):
    reader = _read(generate_plan_pdf(sample_plan))

    uris = {
        str(annotation.get_object().get("/A", {}).get("/URI"))
        for page in reader.pages
        for annotation in page.get("/Annots", [])
    }

    assert {
        "https://example.com/oats",
        "https://example.com/salad",
        "https://example.com/salmon",
    } <= uris


def test_does_not_export_private_fields(sample_plan):
    pdf = generate_plan_pdf(sample_plan)

    assert b"SECRET_SHOULD_NOT_APPEAR" not in pdf
    assert b"PRIVATE_CHAT_SHOULD_NOT_APPEAR" not in pdf

    text = " ".join(
        page.extract_text() for page in _read(pdf).pages
    )

    assert "SECRET_SHOULD_NOT_APPEAR" not in text
    assert "PRIVATE_CHAT_SHOULD_NOT_APPEAR" not in text


def test_html_is_escaped_and_unsafe_urls_are_not_linked(sample_plan):
    sample_plan["days"]["Day 1"]["meals"]["breakfast"] = {
        "title": "Eggs <b> & Toast",
        "calories": 400,
        "url": "javascript:alert(1)",
    }

    reader = _read(generate_plan_pdf(sample_plan))
    text = " ".join(
        page.extract_text() for page in reader.pages
    )

    assert "Eggs <b> & Toast" in text

    uris = [
        str(annotation.get_object().get("/A", {}).get("/URI"))
        for page in reader.pages
        for annotation in page.get("/Annots", [])
    ]

    assert not any(uri.startswith("javascript:") for uri in uris)


@pytest.mark.parametrize(
    "invalid_plan",
    [{}, {"days": {}}, {"days": None}],
)
def test_rejects_missing_or_empty_days(invalid_plan):
    with pytest.raises(ValueError, match="at least one day"):
        generate_plan_pdf(invalid_plan)


def test_rejects_malformed_day():
    with pytest.raises(ValueError, match="Missing meals"):
        generate_plan_pdf({"days": {"Day 1": {}}})


def test_supports_missing_optional_targets_and_url():
    data = generate_plan_pdf(
        {
            "days": {
                "Day 1": {
                    "meals": {
                        "snack": {
                            "title": "Apple",
                            "calories": 95,
                        }
                    }
                }
            }
        }
    )

    assert "Apple" in _read(data).pages[0].extract_text()