"""Control a support RAG workflow based on evidence quality."""

import argparse
import json


SCENARIOS = {
    "security_alert": [[
        {
            "source": "account-security.md",
            "claim": "Revoke active sessions, reset the password, and contact the security team.",
            "score": 0.93,
            "current": True,
        }
    ]],
    "delivery_delay": [
        [{
            "source": "community-post.txt",
            "claim": "Wait another week before contacting support.",
            "score": 0.39,
            "current": True,
        }],
        [{
            "source": "delivery-support.md",
            "claim": "Start a carrier investigation after 48 hours without a scan.",
            "score": 0.88,
            "current": True,
        }],
    ],
    "returns_conflict": [[
        {
            "source": "returns-policy.md",
            "claim": "Unopened accessories can be returned within 30 days.",
            "score": 0.91,
            "current": True,
        },
        {
            "source": "store-notice.md",
            "claim": "Unopened accessories can be returned within 14 days.",
            "score": 0.89,
            "current": True,
        },
    ]],
    "unknown_product": [[], []],
    "retired_article": [[
    {
        "source": "old-device-guide.md",
        "claim": "Ask the customer to reset the device to factory settings.",
        "score": 0.92,
        "current": False,
    }
]],
}


def assess_evidence(evidence: list[dict]) -> str:
    """Return the evidence status used by the routing policy."""
    if not evidence:
        return "empty"

    if any(not item.get("current", False) for item in evidence):
        return "stale"

    strong_claims = {
        item["claim"].strip().lower()
        for item in evidence
        if item.get("score", 0.0) >= 0.75
    }

    if len(strong_claims) > 1:
        return "conflicting"

    if max(item.get("score", 0.0) for item in evidence) >= 0.75:
        return "strong"

    return "weak"


def choose_action(evidence_status: str, attempts: int) -> str:
    """Choose a bounded action from the evidence and previous attempts."""
    if evidence_status == "strong":
        return "answer"

    # if evidence_status in {"conflicting", "stale"}:
    #     return "escalate"

    if evidence_status == "conflicting":
        return "escalate"

    if evidence_status == "stale": # stale means no longer fresh
        return "rewrite" if attempts == 0 else "escalate"

    if attempts == 0:
        return "retrieve" if evidence_status == "empty" else "rewrite"

    return "escalate"


def run_policy(scenario: str) -> dict:
    """Run a bounded evidence-based retrieval workflow."""
    batches = SCENARIOS[scenario]
    attempts = 0
    evidence = batches[0]
    trace = []

    while attempts <= 2:
        evidence_status = assess_evidence(evidence)
        action = choose_action(evidence_status, attempts)

        trace.append(
            {
                "evidence_status": evidence_status,
                "action": action,
            }
        )

        if action == "answer":
            best = max(evidence, key=lambda item: item["score"])

            return {
                "status": "completed",
                "answer": best["claim"],
                "sources": [best["source"]],
                "trace": trace,
            }

        if action in {"retrieve", "rewrite"}:
            attempts += 1
            evidence = batches[min(attempts, len(batches) - 1)]
            continue

        return {
            "status": "needs_human",
            "answer": None,
            "sources": [],
            "trace": trace,
        }

    return {
        "status": "needs_human",
        "answer": None,
        "sources": [],
        "trace": trace,
    }


def main() -> None:
    """Run an evidence-policy scenario from the command line."""
    parser = argparse.ArgumentParser(
        description="Run an evidence-based RAG policy"
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="delivery_delay",
    )
    args = parser.parse_args()

    print(json.dumps(run_policy(args.scenario), indent=2))


if __name__ == "__main__":
    main()