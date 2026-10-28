"""Shared contracts and configuration for the travel agent team."""

import operator
from typing import Annotated, Any, TypedDict


MODEL = "gpt-5-mini"


class AgentProtocolError(RuntimeError):
    """Raised when a model returns an unsafe or malformed tool request."""


class TravelState(TypedDict, total=False): # total = False indicates that every key may be omitted!
    case_id: str
    request: str
    client: Any
    findings: Annotated[list[dict], operator.add]   # Annotate adds extra information to a type[""].
                                                    # Here it Here, for lists, operator.add performs concatenation
                                                    # so, operator.add(["A"], ["B"]) = ["A", "B"]
    trace: Annotated[list[str], operator.add]       # LandGraph interprets operator.add as a function, that combines the existing value with a node's update
    plan: str
    status: str


AGENTS = {
    "transport": {
        "tool": "lookup_transport",
        "description": "Look up verified transport options for a travel case.",
        "role": "You are a transport planning agent. Use only your tool evidence.",
    },
    "weather": {
        "tool": "lookup_weather",
        "description": "Look up verified weather guidance for a travel case.",
        "role": "You are a weather planning agent. Use only your tool evidence.",
    },
    "hotel": {
        "tool": "lookup_hotel",
        "description": "Look up verified hotel information for a travel case.",
        "role": "You are a hotel planning agent. Use only your tool evidence.",
    },
}
