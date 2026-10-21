"""Change one question and compare a fixed chain with an agent decision."""

from agent_choices import run_agent, run_chain


QUESTION = "When is the project deadline?"

MAX_STEPS = 3


def main() -> None:
    """Print concise chain and agent outcomes for the selected question."""
    chain = run_chain(QUESTION)
    agent = run_agent(QUESTION, max_steps=MAX_STEPS)
    result = run_agent(
        "When is the project deadline?",
        max_steps=1,
    )

    expected_status = "step_limit"

    print("Actual status:", result["status"])
    print("Expected status:", expected_status)
    print("Observed:", result["status"] == expected_status)
    print("Trace:", result["trace"])

    result = run_agent("   ")

    print("Status:", result["status"])
    print("Trace:", result["trace"])

    print(
        "Requests clarification:",
        result["status"] == "needs_clarification",
    )
    result = run_agent("Can you change my password?")

    print("Status:", result["status"])
    print("Actions:", [
        event["action"]
        for event in result["trace"]
    ])

    print(
        "No tool was called:",
        result["trace"] == [],
    )
    print("Question:", QUESTION)
    print("Chain status:", chain["status"])
    print("Chain steps:", chain["trace"])
    print("Agent status:", agent["status"])
    print("Agent actions:", [event["action"] for event in agent["trace"]])


if __name__ == "__main__":
    main()
