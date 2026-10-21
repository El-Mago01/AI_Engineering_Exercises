TRANSCRIPTS = [
    {
        "id": "trace-001",
        "description": "Clear heating issue, warm agent",
        "expected_department": "energy-heating",
        "actual_department": "energy-heating",
        "turns": [
            {"role": "tenant", "text": "My heating has been broken for two days."},
            {"role": "agent", "text": "I am sorry you are dealing with that. I will route you to Energy and Heating."},
        ],
        "hallucination_free": 4,
    },
    {
        "id": "trace-003",
        "description": "Noise complaint, routed incorrectly",
        "expected_department": "tenant-complaints",
        "actual_department": "repairs-maintenance",
        "turns": [
            {"role": "tenant", "text": "My upstairs neighbour is making noise constantly."},
            {"role": "agent", "text": "I understand. I am sending you to Repairs and Maintenance."},
        ],
        "hallucination_free": 4,
    },
    {
        "id": "trace-004",
        "description": "Loop, then invented department",
        "expected_department": "energy-heating",
        "actual_department": "energy-heating",
        "turns": [
            {"role": "tenant", "text": "My hot water is not working."},
            {"role": "agent", "text": "Can you tell me more about the issue?"},
            {"role": "tenant", "text": "The hot water stopped yesterday."},
            {"role": "agent", "text": "Can you tell me more about the issue?"},
            {"role": "agent", "text": "I will route you to our Plumbing and Utilities Emergency Response Centre."},
        ],
        "hallucination_free": 1,
    },
]


def rule_based_eval(transcript):
    agent_messages = [
        turn["text"].strip().lower()
        for turn in transcript["turns"]
        if turn["role"] == "agent"
    ]
    return {
        "routing_correct": transcript["actual_department"] == transcript["expected_department"],
        "turn_count": len(transcript["turns"]),
        "has_loop": len(agent_messages) != len(set(agent_messages)),
    }


def needs_review(rules, hallucination_free):
    return (
        not rules["routing_correct"]
        or rules["has_loop"]
        or hallucination_free < 3
    )


def main():
    rows = []
    for transcript in TRANSCRIPTS:
        rules = rule_based_eval(transcript)
        rows.append({
            "id": transcript["id"],
            "routing": "pass" if rules["routing_correct"] else "fail",
            "turns": rules["turn_count"],
            "loop": "yes" if rules["has_loop"] else "no",
            "hallucination_free": transcript["hallucination_free"],
            "review": "yes" if needs_review(rules, transcript["hallucination_free"]) else "no",
        })

    print("CONVERSATION QUALITY CHECK")
    print("ID        ROUTING  TURNS  LOOP  HALLUCINATION_FREE  REVIEW")
    print("------------------------------------------------------------")
    for row in rows:
        print(
            f"{row['id']}  {row['routing']:<7}  {row['turns']:<5}  "
            f"{row['loop']:<4}  {row['hallucination_free']:<18}  {row['review']}"
        )

    print("\nSUMMARY")
    print(f"Cases checked: {len(rows)}")
    print(f"Routing failures: {sum(row['routing'] == 'fail' for row in rows)}")
    print(f"Loop failures: {sum(row['loop'] == 'yes' for row in rows)}")
    print(f"Hallucination failures: {sum(row['hallucination_free'] < 3 for row in rows)}")
    print(f"Needs human review: {sum(row['review'] == 'yes' for row in rows)}")


if __name__ == "__main__":
    main()
