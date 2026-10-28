"""Coordinator node for the travel agent team."""

from team_state import MODEL, TravelState


def coordinator_node(state: TravelState) -> dict:
    print(f"coordinator_node(state = {state})")
    """Create one cited plan after all specialist branches have joined."""
    completed = [item for item in state["findings"] if item["status"] == "completed"]
    missing = [item["agent"] for item in state["findings"] if item["status"] == "failed"]

    if not completed:
        return {
            "plan": f"No verified evidence available.\nMissing evidence: {', '.join(missing)}",
            "status": "needs_human",
            "trace": ["coordinator escalated without a model call"],
        }

    # Preserve citations, append missing evidence, and return the correct
    # status plus one coordinator trace entry.

    response = state["client"].responses.create(
        model=MODEL,
        instructions=(
            "Create a short practical travel plan using only these findings. "
            "Keep every citation and add no facts."
        ),
        input="\n".join(item["finding"] for item in completed),
    )
    plan = response.output_text.strip()
    print(plan)
    """
    Start with response.output_text.strip(). When evidence is missing, append:
    
    plan += f"\nMissing evidence: {', '.join(missing)}"
    Return completed only when no evidence is missing. Otherwise return needs_human. 
    Add one trace entry showing that the coordinator combined the findings.
    """
    if missing:
        plan += f"\nMissing evidence: {', '.join(missing)}"

    return {
        "plan": plan,
        "status": "needs_human" if missing else "completed",
        "trace": ["Coordinator combined the findings."],
    }
