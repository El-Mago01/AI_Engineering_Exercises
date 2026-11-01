"""Build a LangGraph workflow for routing customer-support tickets."""

import argparse
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class TicketState(TypedDict):
    """Values shared by every node during one ticket run."""

    ticket: str
    category: str
    article: str
    status: str
    reply: str
    trace: list[str]


ARTICLES = {
    "delivery": "Open a carrier investigation after 48 hours without a tracking scan.",
    "technical": "Record the export job ID and retry the export once.",
}


def classify_ticket(state: TicketState) -> dict:
    """Classify the ticket and record the node transition."""
    ticket = state["ticket"].lower()
    if "login" in ticket or "charged" in ticket:
        category = "sensitive"
    elif "delivery" in ticket or "tracking" in ticket:
        category = "delivery"
    elif "export" in ticket or "csv" in ticket:
        category = "technical"
    elif "invoice" in ticket or "charged" in ticket:
      category = "billing"
    else:
        category = "unknown"
    return {"category": category, "trace": [*state["trace"], "classify"]}


def retrieve_article(state: TicketState) -> dict:
    """Retrieve the support article for the classified category."""
    article = ARTICLES.get(state["category"], "")
    return {"article": article, "trace": [*state["trace"], "retrieve"]}


def choose_route(state: TicketState) -> str:
    """Route answerable tickets to drafting and risky tickets to review."""
    if state["category"] == "sensitive" or not state["article"]:
        return "human_review"
    return "draft_reply"


def draft_reply(state: TicketState) -> dict:
    """Create a grounded reply from the retrieved article."""
    return {
        "status": "completed",
        "reply": state["article"],
        "trace": [*state["trace"], "draft_reply"],
    }


def human_review(state: TicketState) -> dict:
    """Stop safely when the ticket requires judgement or permission."""
    return {
        "status": "needs_human",
        "reply": "Send this ticket to a support specialist.",
        "trace": [*state["trace"], "human_review"],
    }


def build_graph() -> object:
    """Compile the ticket-routing nodes and edges into a runnable graph."""
    graph = StateGraph(TicketState)
    graph.add_node("classify", classify_ticket)
    graph.add_node("retrieve", retrieve_article)
    graph.add_node("draft_reply", draft_reply)
    graph.add_node("human_review", human_review)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        choose_route,
        {"draft_reply": "draft_reply", "human_review": "human_review"},
    )
    graph.add_edge("draft_reply", END)
    graph.add_edge("human_review", END)
    return graph.compile()


def run_ticket(ticket: str) -> TicketState:
    """Invoke the compiled graph with a complete initial state."""
    initial_state: TicketState = {
        "ticket": ticket,
        "category": "",
        "article": "",
        "status": "running",
        "reply": "",
        "trace": [],
    }
    return build_graph().invoke(initial_state)


def main() -> None:
    """Run one ticket through the compiled LangGraph workflow."""
    parser = argparse.ArgumentParser(description="Route a customer-support ticket")
    parser.add_argument(
        "ticket",
        nargs="?",
        default="My parcel tracking has not changed for three days.",
    )
    args = parser.parse_args()
    result = run_ticket(args.ticket)
    print("Category:", result["category"])
    print("Route:", " -> ".join(result["trace"]))
    print("Status:", result["status"])
    print("Reply:", result["reply"])


if __name__ == "__main__":
    main()
