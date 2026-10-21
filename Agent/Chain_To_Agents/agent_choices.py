"""Compare a fixed chain with a bounded agent loop."""

import argparse
import json


DEADLINE = "Project deadline: Friday 17:00"
COURSE_STATUS = "Course status: active"

def run_chain(question: str) -> dict:
    """Run a developer-chosen sequence, whether or not retrieval is useful."""
    if not question.strip():
        return {"status": "invalid", "answer": None, "trace": ["validate:empty"]}
    return {
        "status": "completed",
        "answer": DEADLINE if "deadline" in question.lower() else "No matching answer",
        "trace": ["validate", "lookup_deadline", "format_answer"],
    }


def choose_action(question: str, observations: list[str]) -> str:
    """Choose the next permitted action from the question and observations."""
    text = question.lower()
    if "deadline" in text and not observations:
        return "lookup_deadline"
    if "deadline" in text and observations:
        return "finish"
    if "course active" in text and not observations:
        return "lookup_status"
    if "course active" in text and observations:
        return "finish"
    return "escalate"


def run_agent(question: str, max_steps: int = 3) -> dict:
    """Run a bounded decision/action/observation loop."""
    if not question.strip():
        return {"status": "needs_clarification", "answer": None, "trace": []}
    if max_steps < 1:
        raise ValueError("max_steps_must_be_positive")

    observations: list[str] = []
    trace: list[dict] = []
    for step in range(1, max_steps + 1):
        action = choose_action(question, observations)
        event = {"step": step, "action": action}
        if action == "lookup_deadline":
            observations.append(DEADLINE)
            event["observation"] = DEADLINE
            trace.append(event)
            continue
        if action == "finish":
            trace.append(event)
            return {"status": "completed", "answer": observations[-1], "trace": trace}
        if action == "lookup_status":
            observations.append(COURSE_STATUS)
            event["observation"] = COURSE_STATUS
            trace.append(event)
            continue
        trace.append(event)
        return {
            "status": "needs_human",
            "answer": "Route to student support",
            "trace": trace,
        }

    return {"status": "step_limit", "answer": None, "trace": trace}


def main() -> None:
    """Run the chain or agent example from the command line."""
    parser = argparse.ArgumentParser(
        description="Compare a fixed chain with a bounded agent.",
        epilog="Example: python main.py --mode agent --max-steps 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter

    )
    parser.add_argument("--mode", choices=["chain", "agent"], default="agent", metavar="'chain' or 'agent'")
    parser.add_argument("--question", default="Is my course active?", metavar="TEXT")
    parser.add_argument("--max-steps", type=int, default=3, metavar="INTEGER")
    args = parser.parse_args()
    if args.mode == "chain":
        result = run_chain(args.question)
    else:
        result = run_agent(args.question, args.max_steps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
