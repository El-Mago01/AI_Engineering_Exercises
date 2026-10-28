"""Use GPT-5 mini inside a real LangGraph ticket-routing node."""

from typing import TypedDict
import os
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("API_KEY")


MODEL = "gpt-5-mini"
client = OpenAI(api_key=api_key)


class TriageState(TypedDict):
    """State shared by the live ticket-triage graph."""

    ticket: str
    category: str
    outcome: str


def classify_with_model(state: TriageState) -> dict:
    """Classify a real ticket into one of three allowed categories."""
    response = client.responses.create(
        model=MODEL,
        instructions="Reply with only DELIVERY, TECHNICAL, or SENSITIVE.",
        input=state["ticket"],
    )
    return {"category": response.output_text.strip().upper()}


def choose_route(state: TriageState) -> str:
    """Keep sensitive actions behind a human-review boundary."""
    return "human_review" if state["category"] == "SENSITIVE" else "queue_support"


def queue_support(state: TriageState) -> dict:
    """Record a normal support-queue outcome."""
    return {"outcome": f"Queued for {state['category'].lower()} support"}


def human_review(state: TriageState) -> dict:
    """Record a review outcome for security or billing risk."""
    return {"outcome": "Human review required before any account action"}


def build_graph() -> object:
    """Compile the model-backed ticket-routing graph."""
    graph = StateGraph(TriageState)
    graph.add_node("classify", classify_with_model)
    graph.add_node("queue_support", queue_support)
    graph.add_node("human_review", human_review)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        choose_route,
        {"queue_support": "queue_support", "human_review": "human_review"},
    )
    graph.add_edge("queue_support", END)
    graph.add_edge("human_review", END)
    return graph.compile()


def main() -> None:
    """Route an unfamiliar-login report through LangGraph."""
    result = build_graph()
    result=result.invoke(
        {
            "ticket": "I received a login alert from a device I do not recognise.",
            "category": "",
            "outcome": "",
        }
    )
    print("Model:", MODEL)
    print("Category:", result["category"])
    print("Outcome:", result["outcome"])


if __name__ == "__main__":
    main()
