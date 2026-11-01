"""A small LangGraph team: two specialists and one supervisor."""

from langgraph.graph import END, START, StateGraph


RECORDS = {
    "delivery": {
        "source_id": "DEL-104",
        "claim": "Eighteen of twenty orders arrived on time.",
    },
    "compliance": {
        "source_id": "CMP-208",
        "claim": "One packaging-control issue is still open.",
    },
}


def specialist(category: str) -> dict:
    """Return one small, predictable result from a specialist."""
    evidence = RECORDS.get(category)
    if evidence is None:
        return {"specialist": category, "status": "failed", "error": "no evidence"}
    return {"specialist": category, "status": "completed", "evidence": evidence}


def delivery_node(state: dict) -> dict:
    """Ask the delivery specialist for evidence."""
    return {**state, "results": [*state["results"], specialist("delivery")]}


def compliance_node(state: dict) -> dict:
    """Ask the compliance specialist for evidence."""
    return {**state, "results": [*state["results"], specialist("compliance")]}


def supervisor_node(state: dict) -> dict:
    """Combine both specialist results into one final answer."""
    lines = ["Supplier briefing"]
    missing = []
    for result in state["results"]:
        if result["status"] == "completed":
            evidence = result["evidence"]
            lines.append(f"- {evidence['claim']} [{evidence['source_id']}]")
        else:
            missing.append(result["specialist"])
    if missing:
        lines.append(f"- Missing evidence: {', '.join(missing)}")
    return {
        **state,
        "brief": "\n".join(lines),
        "status": "needs_human" if missing else "completed",
    }


def build_graph() -> object:
    """Connect the two specialists to the final supervisor."""
    graph = StateGraph(dict)
    graph.add_node("delivery", delivery_node)
    graph.add_node("compliance", compliance_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_edge(START, "delivery")
    graph.add_edge("delivery", "compliance")
    graph.add_edge("compliance", "supervisor")
    graph.add_edge("supervisor", END)
    return graph.compile()


def run_team() -> dict:
    """Run the complete team with an empty shared result list."""
    return build_graph().invoke({"results": []})


if __name__ == "__main__":
    outcome = run_team()
    print(outcome["brief"])
    print(f"Status: {outcome['status']}")
