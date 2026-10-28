"""
This rather a stupid exercise. Just ignore this one
If the ASR mishears “turn on the light” as “turn on the wifi” or something similar, everything downstream can go weird even if the later stages are perfect.

Let’s now turn this into a real mini pipeline in Python inside the editor.


We’ll build this first version:


Simulated “speech input” = a fixed string.
“Transcription” = we’ll just pass that string through (no real ASR yet).
Simple retrieval: pick from a tiny list of docs about the light.
“Reasoning”: decide whether to call set_light(True) or set_light(False) based on words in the transcript.
Execute set_light.
Print a final “spoken” response.
Your turn:


Open the code editor for this lesson and create a new script that:


Defines a function set_light(on: bool) that returns:
"Light is now ON." if on is True
"Light is now OFF." if on is False
Calls set_light(True) once and prints its return value.

"""

def set_light(on: bool) -> None:
    if on:
        return "Light is now ON"
    else:
        return "Light is now OFF"



def run_agent(transcript: str) -> str:
    retrieved_docs = []

    if "on" in transcript.lower() and "light" in transcript.lower():
        on_value=True
        spoken_response = "Okay, I turned on the light!"
        tool_result = set_light(on_value)
    else:
        if "light" in transcript.lower():
            on_value=False
            spoken_response = "Okay, I turned off the light!"
            tool_result = set_light(on_value)
        else:
            on_value="tool not available"
            tool_result = "tool not available"
            spoken_response = "sorry, I can only control the light"
    if not retrieved_docs:
        on_value = ""
        tool_result = "no tool execution done"
        spoken_response = "Not enough background information available"
    order = {
        "transcript": transcript,
        "retrieved_docs": retrieved_docs,
        "tool_call": {"name": "set_light", "args": {"on": on_value}},
        "tool_result": tool_result,
        "spoken_response": spoken_response,
    }
    return order

def main() -> None:
    # result = set_light(on=True)
    result = run_agent("please turn on the light in the room")
    print(result)


if __name__ == "__main__":
    main()