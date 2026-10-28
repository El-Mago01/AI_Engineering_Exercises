"""Scoped evidence tools for the travel-planning agents."""

import json
from pathlib import Path


DATA_PATH = Path(__file__).parent / "data" / "travel_cases.json"
CASES = json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _lookup(case_id: str, category: str) -> dict:
    if case_id not in CASES:
        raise ValueError("unknown_case")
    record = CASES[case_id].get(category)
    if record is None:
        return {"ok": False, "category": category, "error": "source_unavailable"}
    return {"ok": True, "category": category, **record}


def lookup_transport(case_id: str) -> dict:
    print(f"lookup_transport(case_id= {case_id})")
    """Return raw transport evidence for one travel case."""
    return _lookup(case_id, "transport")


def lookup_weather(case_id: str) -> dict:
    print(f"lookup_weather(case_id= {case_id})")

    """Return raw weather evidence for one travel case."""
    return _lookup(case_id, "weather")


def lookup_hotel(case_id: str) -> dict:
    print(f"lookup_hotel(case_id= {case_id})")

    """Return raw hotel evidence for one travel case."""
    return _lookup(case_id, "hotel")


TOOL_REGISTRY = {
    "lookup_transport": lookup_transport,
    "lookup_weather": lookup_weather,
    "lookup_hotel": lookup_hotel,
}


def tool_schema(name: str, description: str) -> dict:
    """Build one Responses API function-tool schema."""
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"case_id": {"type": "string"}},
            "required": ["case_id"],
            "additionalProperties": False,
        },
        "strict": True,
    }
