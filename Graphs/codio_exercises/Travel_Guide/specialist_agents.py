"""Tool-calling transport, weather, and hotel agent nodes."""

import json

from openai import OpenAI
import os

from team_state import AGENTS, MODEL, AgentProtocolError, TravelState
from tools import TOOL_REGISTRY, tool_schema
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def _function_call(response: object) -> object:
    calls = [item for item in response.output if item.type == "function_call"]
    if len(calls) != 1:
        raise AgentProtocolError("expected_one_tool_call")
    return calls[0]


def run_specialist_agent(client: OpenAI, case_id: str, request: str, agent: str) -> dict:
    print(f"run_specialist_agent client={client}, case_id={case_id}, request={request}, agent={agent}")
    """Run one scoped tool-calling specialist and return its cited finding."""
    config = AGENTS[agent]
    expected_tool = config["tool"]

    # and safely parse the case_id from the JSON arguments.
    first = client.responses.create(
        model=MODEL,
        instructions=f"{config['role']} Call your required tool for the supplied case.",
        input=f"Case ID: {case_id}\nTravel request: {request}",
        tools=[tool_schema(expected_tool, config["description"])],
        tool_choice="required",
    )
    call = _function_call(first)
    if call.name != expected_tool:
        return {
            "agent": agent,
            "tool": expected_tool,
            "status": status,
            "error": f"agent {agent} used an unexpected tool {call.name}",
        }
    try:
        args = json.loads(call.arguments)
    except json.decoder.JSONDecodeError:
        raise AgentProtocolError("invalid_tool_arguments")
    # evidence is unavailable. Otherwise send function_call_output to the
    # model and return a completed, cited finding.
    tool_result = TOOL_REGISTRY[expected_tool](case_id)
    if not tool_result["ok"]:
        return {
            "agent": agent,
            "tool": expected_tool,
            "status": "failed",
            "error": tool_result["error"],
        }
    second = client.responses.create(
        model=MODEL,
        previous_response_id=first.id,
        instructions=(
            "Return one short finding. Include the supplied source ID "
            "in square brackets and add no facts."
        ),
        input=[{
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(tool_result),
        }],
    )
    contract = second.output_text.strip()
    return {
        "agent": agent,
        "tool": expected_tool,
        "status": "completed",
        "source_id": tool_result["source_id"],
        "finding": contract,
    }

def _worker(state: TravelState, agent: str) -> dict:
    print(f"_worker(state = {state}, agent = {agent})")

    result = run_specialist_agent(
        state["client"], state["case_id"], state["request"], agent
    )
    return {
        "findings": [result],
        # "status": "completed",
        "trace": [f"{agent} agent used {AGENTS[agent]['tool']}"],
    }


def transport_node(state: TravelState) -> dict:
    print(f"transport_node(state = {state})")
    return _worker(state, "transport")


def weather_node(state: TravelState) -> dict:
    print(f"weather_node(state = {state})")
    return _worker(state, "weather")


def hotel_node(state: TravelState) -> dict:
    print(f"hotel_node(state = {state})")
    return _worker(state, "hotel")

def main():
    client = OpenAI(api_key=api_key)
    case_id = "berlin-conference"
    request = "Find a safe, practical way to reach the conference and keep tonight's booking."
    agent = "transport"
    state: TravelState = {
        "case_id": case_id,
        "request": request,
        "client": client,
        "findings": [],
        "trace": [],
        "plan": "",
        "status": "",
    }


    result = transport_node(state)
    print(result)


if __name__ == "__main__":
    main()
