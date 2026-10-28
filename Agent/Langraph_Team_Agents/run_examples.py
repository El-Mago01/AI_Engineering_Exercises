"""Run a real OpenAI-backed team with two specialists and one supervisor."""

from openai import OpenAI
from langgraph.graph import END, START, StateGraph
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-5-mini"
client = OpenAI(api_key=api_key)


def ask(role: str, evidence: str) -> str:
    """Make one real model call for one clear team role."""
    response = client.responses.create(
        model=MODEL,
        instructions=f"You are the {role}. Be factual and use only the supplied evidence.",
        input=evidence,
    )
    return response.output_text.strip()


def delivery_node(state: dict) -> dict:
    finding = ask("delivery specialist", "18 of 20 orders arrived on time. Source: DEL-104.")
    return {**state, "delivery": finding}


def compliance_node(state: dict) -> dict:
    finding = ask("compliance specialist", "One packaging-control issue remains open. Source: CMP-208.")
    return {**state, "compliance": finding}


def supervisor_node(state: dict) -> dict:
    evidence = f"Delivery finding:\n{state['delivery']}\n\nCompliance finding:\n{state['compliance']}"
    final = ask("supervisor", f"Combine these findings into one short cited briefing.\n\n{evidence}")
    return {**state, "final": final, "status": "completed"}


def main() -> None:
    graph = StateGraph(dict)
    graph.add_node("delivery", delivery_node)
    graph.add_node("compliance", compliance_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_edge(START, "delivery")
    graph.add_edge("delivery", "compliance")
    graph.add_edge("compliance", "supervisor")
    graph.add_edge("supervisor", END)
    result = graph.compile().invoke({})
    print(f"Model: {MODEL}")
    print(f"Delivery specialist: {result['delivery']}")
    print(f"Compliance specialist: {result['compliance']}")
    print(f"Supervisor: {result['final']}")


if __name__ == "__main__":
    main()
