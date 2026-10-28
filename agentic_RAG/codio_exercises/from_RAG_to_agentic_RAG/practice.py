"""Change the evidence scenario and inspect the agentic RAG policy."""

from agentic_rag import SCENARIOS, run_policy


SCENARIO = "delivery_delay"


def main() -> None:
    """Print the evidence decisions without dumping complete documents."""
    if SCENARIO not in SCENARIOS:
        print("Available scenarios:", list(SCENARIOS))
        return
    result = run_policy(SCENARIO)
    print("Scenario:", SCENARIO)
    print("Actions:", [event["action"] for event in result["trace"]])
    print("Status:", result["status"])
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])


if __name__ == "__main__":
    main()
