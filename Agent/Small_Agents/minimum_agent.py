"""
This minimum agent, uses 1 tool to check the status of GitHub and was established within the
online session from Slawa Gladkov on 22.10 to get awareness of agentic SW..


"""

import json
import requests
from pathlib import Path
from openai import OpenAI
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
)
from openai.types.responses.response_input_param import FunctionCallOutput

import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

api_key = os.getenv("OPENAI_API_KEY")

def get_github_status():
    try:
        github_response = requests.get(
            "https://www.githubstatus.com/api/v2/status.json",
            timeout=10
        )
        github_response.raise_for_status()
        data = github_response.json()
        return data["status"]["description"]

    except (requests.RequestException, KeyError, ValueError):
        return "Github's current status could not be verified"

print(get_github_status())

TOOLS: list[FunctionToolParam] = [
    {
        "type": "function",
        "name": "get_github_status",
        "description": "Get Github's current operational status Use this when the user asks about "
                       "GitHub's current availability",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    "strict": True,
    }]

client = OpenAI(api_key=api_key)

question = "Is GitHub operative right now?"

response = client.responses.create(
    model="gpt-5-mini",
    instructions = "Answer briefly and accurately, Use the available tool when current GitHub status "
                   "information is required",
    input = question,
    tools = TOOLS,
    stream=False,

)
print(response.output[:50])

tool_call = None

for item in response.output:
    if isinstance(item, ResponseFunctionToolCall):
        tool_call = item
        break

tool_result = None
print("Tool call is: ", tool_call.name)
if tool_call is None:
    print("The model did not select a tool.")

elif tool_call.name == "get_github_status":
    print(f"Selected tool: {tool_call.name}")
    tool_result = get_github_status()
    print(f"Tool result: {tool_result}")

else:
    print(f"Tool not permitted: {tool_call.name}")


if tool_result is not None and tool_call is not None:
    final_response = client.responses.create(
        model="gpt-5-mini",
        previous_response_id=response.id,
        instructions=(
            "Answer the original question briefly. "
            "Use only the result returned by the tool."
        ),
        tools=TOOLS,
        input=[
            FunctionCallOutput(
                type="function_call_output",
                call_id=tool_call.call_id,
                output=tool_result,
            )
        ],
    )

    print(f"Final answer: {final_response.output_text}")

