"""Public entry point for the Travel Planning Agent Team."""

from openai import OpenAI
import os
from graph_builder import build_graph
from team_state import AgentProtocolError
from tools import CASES
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("API_KEY")
client = OpenAI(api_key=api_key)


def run_team(case_id: str, client: OpenAI | None = None) -> dict:
    """Run the live travel-planning team for one case."""
    print(f"run_team(case_id={case_id}, client={client} )")
    if case_id not in CASES:
        raise ValueError("unknown_case")
    case = CASES[case_id]
    result = build_graph().invoke({
        "case_id": case_id,
        "request": case["request"],
        "client": client or OpenAI(),
        "findings": [],
        "trace": [],
    })
    print(f"result of run_team = {result}")
    result.pop("client", None)
    return result
