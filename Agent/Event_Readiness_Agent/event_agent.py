"""Command-line agent for checking event readiness with local tools.

Complete the bounded Responses API loop in ``run_agent`` after implementing the
local tools in ``event_tools.py``.
"""

import argparse
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from event_tools import DEFAULT_CONTEXT, ToolContext, dispatch, load_json

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-5-mini"
MAX_TOOL_CALLS = 4

SYSTEM_INSTRUCTIONS = """
You are an event-operations assistant.

Use the supplied tools for event facts and readiness checks.
Before creating a setup request, explain why it is needed.
Report tool errors accurately and never claim that a failed action succeeded.

If a tool returns event_not_found, report that the event is unknown and stop.
Do not call another event tool for the same unknown event.

Answer the current request directly.
Do not ask follow-up questions.
Do not offer actions that are not supported by the supplied tools.
Do not invent operational facts or recommendations.
End after giving the answer.
""".strip()

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "get_event_details",
        "description": "Get the stored details for one event.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "An event identifier such as EVT-101.",
                }
            },
            "required": ["event_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "check_event_readiness",
        "description": "Check venue capacity and required equipment for an event.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "An event identifier such as EVT-101.",
                }
            },
            "required": ["event_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "create_setup_request",
        "description": "Create an operator-approved request for an event that is not ready.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "An event identifier such as EVT-102.",
                }
            },
            "required": ["event_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def run_agent(
    user_request: str,
    api_client: object | None = None,
    context: ToolContext = DEFAULT_CONTEXT,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> dict:
    """Run a bounded tool-calling conversation for one user request."""

    # 1. Create a client when api_client is None.
    # 2. Send the request and TOOL_DEFINITIONS to the Responses API.
    # 3. Append response.output to the running input.
    # 4. Dispatch each function_call and return its output with the call_id.
    # 5. Record one trace entry per attempted tool call.
    # 6. Return the final text or stop when max_tool_calls is reached.

    client = api_client or OpenAI()
    running_input = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_request},
    ]
    trace = []
    tool_call_count = 0

    while tool_call_count < max_tool_calls:
        response = client.responses.create(
            model=MODEL_NAME,
            input=running_input,
            tools=TOOL_DEFINITIONS,
        )
        running_input.extend(response.output)
        function_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]
        if not function_calls:
            return {
                "status": "completed",
                "answer": response.output_text,
                "trace": trace,
            }
        for call in function_calls:
            if tool_call_count >= max_tool_calls:
                break
            try:
                arguments = json.loads(call.arguments)
            except (TypeError, json.JSONDecodeError):
                arguments = None

            result = dispatch(call.name, arguments, context)
            trace.append({
                "step": tool_call_count,
                "tool": call.name,
                "arguments": arguments,
                "result": result["code"],
            })
            running_input.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result),
            })
            tool_call_count += 1

    return {
    "status": "stopped",
    "answer": f"The agent stopped after {max_tool_calls} tool calls.",
    "trace": trace,
    }

def show_demo_data() -> None:
    """Print the supplied events so the starter can be explored offline."""

    events = load_json(DEFAULT_CONTEXT.events_path)["events"]
    print("EventOps starter data")
    for event_id, event in events.items():
        print(f"- {event_id}: {event['name']} ({event['expected_attendees']} attendees)")


def main() -> None:
    """Run the command-line interface."""

    parser = argparse.ArgumentParser(description="Run the EventOps agent.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Show the supplied events without calling OpenAI.",
    )
    parser.add_argument(
        "--request",
        help="Provide a request directly instead of using the input prompt.",
    )
    args = parser.parse_args()

    if args.demo:
        show_demo_data()
        return

    user_request = args.request or input("What should EventOps check? ").strip()
    result = run_agent(user_request, api_client=OpenAI(api_key=api_key))
    print(f"\n{result['answer']}")
    print("\nTrace:")
    for event in result["trace"]:
        print(event)


if __name__ == "__main__":
    main()
