"""Run a real Responses API tool-calling loop."""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-5-mini"
client = OpenAI(api_key=api_key)

DEADLINE_TOOL = {
    "type": "function",
    "name": "lookup_deadline",
    "description": "Return the assessed project deadline.",
    "parameters": {
        "type": "object",
        "properties": {
            "project": {"type": "string"},
        },
        "required": ["project"],
        "additionalProperties": False,
    },
    "strict": True,
}


def lookup_deadline(project: str) -> dict[str, str]:
    """Return a real local tool result for the requested project."""
    return {"project": project, "deadline": "Friday at 17:00"}


def compact(value: object, limit: int = 140) -> str:
    """Reduce the model output to a short final line."""
    return " ".join(str(value).split())[:limit]


def main() -> None:
    """Let the model select a tool, execute it, and complete the response."""
    first = client.responses.create(
        model=MODEL,
        input="When is the multimodal project due?",
        tools=[DEADLINE_TOOL],
        tool_choice="required",
    )
    call = next(item for item in first.output if item.type == "function_call")
    arguments = json.loads(call.arguments)
    if call.name == "lookup_deadline":
        observation = lookup_deadline(arguments["project"])
        final = client.responses.create(
            model=MODEL,
            previous_response_id=first.id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(observation),
                }
            ],
            instructions="Answer in one short sentence.",
        )
        print(f"Model: {MODEL}")
        print(f"Tool selected: {call.name}")
        print(f"Tool result: {observation['deadline']}")
        print(f"Final answer: {compact(final.output_text)}")
    else:
        print(f"Model: {MODEL}")
        print(f"Non-existing tool was selected: {call.name}")

if __name__ == "__main__":
    main()