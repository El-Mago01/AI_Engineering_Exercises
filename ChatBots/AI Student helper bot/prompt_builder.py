MODE_LABELS = {
    "quiz": "quiz questions",
    "summary": "a summary",
    "plan": "a study plan",
}


def build_prompt(topic: str, mode: str) -> str:
    mode_label = MODE_LABELS.get(mode, "study help")
    return (
        "Role: You are a study assistant, specialized in python. "
        f"Context: You help students to study around this topic: {topic}. "
        f"Give them {mode_label}."
    )
