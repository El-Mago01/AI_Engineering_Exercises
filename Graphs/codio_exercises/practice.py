"""Complete a small LangGraph for routing an export support ticket."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class ExportState(TypedDict):
    """State shared by the export-support nodes."""

    ticket: str
    route: str
    response: str


def inspect_ticket(state: ExportState) -> dict:
    """Choose whether the ticket has enough detail for troubleshooting."""
    has_job_id = "job-" in state["ticket"].lower()
    return {"route": "troubleshoot" if has_job_id else "request_details"}


def troubleshoot(state: ExportState) -> dict:
    """Return the first approved troubleshooting action."""
    return {"response": "Retry the CSV export once and keep the job ID."}


def request_details(state: ExportState) -> dict:
    """Ask for the identifier needed by the support team."""
    return {"response": "Please provide the export job ID."}


def choose_route(state: ExportState) -> str:
    """Return the node name selected during inspection."""
    return state["route"]


def build_graph() -> object:
    """Build and compile the export-support graph."""
    # Add the three nodes and connect START, the conditional route, and END.
    """Build and compile the export-support graph."""
    # Add the three nodes and connect START, the conditional route, and END.
    graph = StateGraph(ExportState)
    graph.add_node("inspect", inspect_ticket)
    graph.add_node("troubleshoot", troubleshoot)
    graph.add_node("request_details", request_details)
    graph.add_edge(START, "inspect")
    graph.add_conditional_edges(
    "inspect",
    choose_route,
    {
        "troubleshoot": "troubleshoot",
        "request_details": "request_details",
    },
    )
    graph.add_edge("troubleshoot", END)
    graph.add_edge("request_details", END)
    return graph.compile()


def main() -> None:
    """Invoke the graph after build_graph has been completed."""
    initial_state: ExportState = {
        "ticket": "CSV export job-4821 is stuck at 100%.",
        "route": "",
        "response": "",
    }
    try:
        result = build_graph().invoke(initial_state)
    except NotImplementedError as error:
        print(error)
        return
    print("Route:", result["route"])
    print("Response:", result["response"])


if __name__ == "__main__":
    main()
