"""
Talking to agents demo.

Teaching goal: a voice interface is not automatically an agent. The agent
behavior comes from memory, state, tool planning, permissions, and
controlled execution.

Pipeline:
  1. Setup & verification     - check env vars, ffmpeg, local speech tools
  2. Audio preprocessing      - convert to 16kHz mono WAV
  3. Transcription (ASR)      - speech-to-text via OpenAI, or fallback text
  4. Conversation state       - track turns, pending actions, memories, results
  5. Semantic memory          - embed a small memory store and the transcript
  6. Retrieval                - rank memories by cosine similarity
  7. Tools                    - plain Python functions the agent may call
  8. Tool plan                - decide whether an action is needed (rule-based
                                 by default, optional LLM planner)
  9. Confirmation              - require human approval before state-changing
                                 tool calls
 10. Tool execution           - run the approved tool
 11. Response generation      - report only what actually happened (template
                                 by default, optional LLM writer)
 12. Text-to-speech           - OpenAI TTS, then local fallback, then text
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from IPython.display import Audio, display
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from demo_ui import step_banner
from media_paths import SUPPORT_CALL_AUDIO, output_dir

# Load .env before reading any os.getenv() defaults below - the config
# constants are evaluated once, at import time, so .env must be loaded here
# rather than inside main() or these overrides would never take effect.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

OUTPUT_DIR = output_dir(__file__)

INPUT_AUDIO = SUPPORT_CALL_AUDIO
CLEAN_AUDIO = OUTPUT_DIR / "sample_agent_request_16k_mono.wav"
TTS_OUTPUT = OUTPUT_DIR / "agent_answer.wav"
TTS_AIFF_OUTPUT = OUTPUT_DIR / "agent_answer.aiff"

STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto").lower()
OPENAI_TTS_MODELS = [
    item.strip()
    for item in os.getenv("OPENAI_TTS_MODELS", "gpt-4o-mini-tts,tts-1,tts-1-hd").split(",")
    if item.strip()
]
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

USE_OPENAI_LLM = os.getenv("USE_OPENAI_LLM", "false").lower() in {"1", "true", "yes"}
AGENT_MODEL = os.getenv("OPENAI_AGENT_MODEL", "gpt-4o-mini")

# The application, not the planner, decides which tool calls are
# state-changing and therefore require human confirmation. A planner's own
# "requires_confirmation" field is informational only and is never trusted
# for the execution decision.
STATE_CHANGING_TOOLS = {"create_support_ticket"}

FALLBACK_TRANSCRIPT = (
    "Hi, I cannot log into my account. "
    "The password reset email never arrives, and I need access today. "
    "Please create a support ticket."
)

MEMORY_ITEMS = [
    {
        "id": "mem_001",
        "type": "previous_call",
        "topic": "password_reset",
        "text": (
            "Customer could not log in because the password reset email was not "
            "arriving. Support created a high-priority access ticket and advised "
            "checking spam and email filters."
        ),
    },
    {
        "id": "mem_002",
        "type": "previous_call",
        "topic": "billing",
        "text": (
            "Customer reported being charged twice for the same subscription "
            "renewal. Billing team issued a refund after confirming duplicate "
            "payment."
        ),
    },
    {
        "id": "mem_003",
        "type": "knowledge_base",
        "topic": "email_delivery",
        "text": (
            "Password reset emails may be delayed by spam filters, corporate "
            "firewalls, expired links, or blocked sender domains."
        ),
    },
    {
        "id": "mem_004",
        "type": "previous_call",
        "topic": "delivery_delay",
        "text": (
            "Customer asked about a delayed package and needed an updated "
            "delivery estimate from the shipping provider."
        ),
    },
    {
        "id": "mem_005",
        "type": "knowledge_base",
        "topic": "microphone_issue",
        "text": (
            "If the microphone is not detected, users should check browser "
            "permissions, operating system privacy settings, and selected "
            "input device."
        ),
    },
]

HELP_ARTICLES = {
    "password_reset": (
        "Ask the customer to check spam, verify the email address, wait a few "
        "minutes, and retry the reset link. If access is urgent, create a "
        "support ticket."
    ),
    "billing": "Check invoices, payment provider status, and refund eligibility.",
    "delivery_delay": "Check shipment status and carrier tracking events.",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# Step 1: Setup and verification
# --------------------------------------------------------------------------

def verify_environment() -> None:
    """Check that required env vars, tooling, and input files are present."""
    print("OpenAI key set:", bool(os.environ.get("OPENAI_API_KEY")))
    print("FFmpeg available:", bool(shutil.which("ffmpeg")))
    print("macOS say available:", bool(shutil.which("say")))
    print("espeak available:", bool(shutil.which("espeak")))
    print("Input audio exists:", INPUT_AUDIO.exists())
    print("Speech-to-text model:", STT_MODEL)
    print("Embedding model:", EMBEDDING_MODEL)
    print("TTS provider:", TTS_PROVIDER)
    print("OpenAI TTS candidates:", OPENAI_TTS_MODELS)
    print("TTS voice:", TTS_VOICE)
    print("Use optional OpenAI LLM planner:", USE_OPENAI_LLM)
    if USE_OPENAI_LLM:
        print("Agent model:", AGENT_MODEL)


# --------------------------------------------------------------------------
# Step 2: Audio preprocessing
# --------------------------------------------------------------------------

def prepare_audio(input_path: Path, output_path: Path) -> Path | None:
    """Convert audio to 16 kHz mono WAV when FFmpeg and input audio are available."""
    if not input_path.exists():
        print("No audio file found. The demo will use the fallback transcript.")
        return None

    if not shutil.which("ffmpeg"):
        print("FFmpeg not found. Using the original audio file.")
        return input_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print("Prepared audio:", output_path)
    return output_path


# --------------------------------------------------------------------------
# Step 3: Transcription (ASR)
# --------------------------------------------------------------------------

def transcribe_or_fallback(client: OpenAI, audio_path: Path | None) -> str:
    """Transcribe audio via OpenAI STT, or fall back to a fixed transcript."""
    if audio_path is None:
        return FALLBACK_TRANSCRIPT

    try:
        with audio_path.open("rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=audio_file,
            )
        return result.text.strip()
    except Exception as exc:
        print("Transcription failed. Using fallback transcript.")
        print(type(exc).__name__, exc)
        return FALLBACK_TRANSCRIPT


# --------------------------------------------------------------------------
# Step 4: Conversation state
# --------------------------------------------------------------------------

def new_conversation_state(user_transcript: str) -> dict:
    """Create the state object that carries the conversation across turns."""
    return {
        "conversation_id": f"conv_{uuid.uuid4().hex[:8]}",
        "customer_id": "cust_demo_001",
        "messages": [
            {
                "role": "user",
                "content": user_transcript,
                "timestamp": utc_now(),
            }
        ],
        "pending_action": None,
        "confirmation": None,
        "retrieved_memories": [],
        "tool_results": [],
    }


# --------------------------------------------------------------------------
# Steps 5-6: Semantic memory and retrieval
# --------------------------------------------------------------------------

def build_memory_frame() -> pd.DataFrame:
    """Return the small stand-in memory store used for retrieval."""
    return pd.DataFrame(MEMORY_ITEMS)



def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def retrieve_memories(
    client: OpenAI, memory_df: pd.DataFrame, transcript: str, top_k: int = 3
) -> list[dict]:
    """Embed the transcript and memory store, rank memories by similarity."""
    memory_embeddings = embed_texts(client, memory_df["text"].tolist())
    query_embedding = embed_texts(client, [transcript])

    scores = cosine_similarity(query_embedding, memory_embeddings)[0]
    scored = memory_df.assign(score=scores).sort_values("score", ascending=False)

    print(scored[["id", "type", "topic", "score"]].to_string(index=False))

    # cosine_similarity preserves the float32 dtype from the embeddings,
    # which json.dumps cannot serialize. Cast to plain floats for state.
    records = scored.head(top_k).to_dict(orient="records")
    for record in records:
        record["score"] = float(record["score"])
    return records


# --------------------------------------------------------------------------
# Step 7: Tools
# --------------------------------------------------------------------------

def search_help_article(topic: str) -> dict:
    """Read-only tool: look up guidance for a topic."""
    return {
        "tool": "search_help_article",
        "topic": topic,
        "article": HELP_ARTICLES.get(topic, "No matching article found."),
    }


def create_support_ticket(
    customer_id: str, issue_type: str, priority: str, summary: str, support_tickets: list[dict]
) -> dict:
    """State-changing tool: create a support ticket. Requires confirmation."""
    ticket_id = f"TCK-{len(support_tickets) + 1:04d}"
    ticket = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "issue_type": issue_type,
        "priority": priority,
        "summary": summary,
        "status": "created",
        "created_at": utc_now(),
    }
    support_tickets.append(ticket)
    return ticket


def build_tools(support_tickets: list[dict]) -> dict:
    """Bind tool functions to the mutable state (support_tickets) they act on."""
    return {
        "search_help_article": lambda topic: search_help_article(topic),
        "create_support_ticket": lambda **kwargs: create_support_ticket(
            support_tickets=support_tickets, **kwargs
        ),
    }


# Each tool's name, description, and JSON parameter schema, in OpenAI's
# function-calling format. Only used by the optional LLM planner
# (USE_OPENAI_LLM=true) so the model's tool choice and arguments are
# enforced by a schema instead of parsed out of freeform text.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_help_article",
            "description": "Look up help-center guidance for a topic. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create a support ticket. State-changing; requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "issue_type": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "standard", "high"]},
                    "summary": {"type": "string"},
                },
                "required": ["customer_id", "issue_type", "priority", "summary"],
            },
        },
    },
]


# --------------------------------------------------------------------------
# Step 8: Tool plan
# --------------------------------------------------------------------------

def rule_based_tool_plan(transcript: str, customer_id: str, retrieved_memories: list[dict]) -> dict:
    """Transparent, deterministic planner. This is the default path.

    The "requires_confirmation" field below is informational only: the
    application enforces confirmation independently via STATE_CHANGING_TOOLS
    and does not trust this planner's opinion (see confirm_and_execute).
    """
    text = transcript.lower()

    password_issue = (
        "password" in text
        or "reset" in text
        or "log in" in text
        or "login" in text
        or "account" in text
    )

    asks_for_ticket = "ticket" in text or "support" in text or "access today" in text

    if password_issue and asks_for_ticket:
        return {
            "tool_name": "create_support_ticket",
            "arguments": {
                "customer_id": customer_id,
                "issue_type": "password_reset",
                "priority": "high",
                "summary": (
                    "Customer cannot log in because the password reset email is "
                    "not arriving and needs access today."
                ),
            },
            "requires_confirmation": True,
            "reason": "The request indicates an urgent password reset issue and asks for support action.",
        }

    if password_issue:
        return {
            "tool_name": "search_help_article",
            "arguments": {"topic": "password_reset"},
            "requires_confirmation": False,
            "reason": "The request is about a password reset issue, but no state-changing action is required yet.",
        }

    return {
        "tool_name": "search_help_article",
        "arguments": {"topic": retrieved_memories[0]["topic"] if retrieved_memories else "password_reset"},
        "requires_confirmation": False,
        "reason": "Use the nearest retrieved topic to look up guidance.",
    }


def llm_tool_plan(client: OpenAI, transcript: str, customer_id: str, retrieved_memories: list[dict]) -> dict:
    """Optional planner. Only used when USE_OPENAI_LLM=true and the API account
    has access to a text-generation model.

    Uses real function-calling against TOOL_SCHEMAS so the tool choice and
    its arguments are enforced by a schema, not parsed out of freeform text.
    The "requires_confirmation" field returned below is derived from
    STATE_CHANGING_TOOLS for logging/comparison only: the application still
    enforces confirmation independently and does not trust this value (see
    confirm_and_execute).
    """
    retrieved_text = "\n".join(
        f"- {item['id']} / {item['topic']}: {item['text']}" for item in retrieved_memories
    )

    system_prompt = (
        "You are a bounded support agent planner. Choose exactly one tool "
        "to call based on the transcript and retrieved memories."
    )
    user_prompt = f"""
User transcript:
{transcript}

Customer ID:
{customer_id}

Retrieved memories:
{retrieved_text}
""".strip()

    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=TOOL_SCHEMAS,
        tool_choice="required",
    )

    message = response.choices[0].message
    if not message.tool_calls:
        raise ValueError("Model did not return a tool call")

    tool_call = message.tool_calls[0]
    tool_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    return {
        "tool_name": tool_name,
        "arguments": arguments,
        "requires_confirmation": tool_name in STATE_CHANGING_TOOLS,
        "reason": message.content or "Selected via OpenAI function calling.",
    }


def build_tool_plan(client: OpenAI, state: dict) -> dict:
    """Decide whether an action is needed. Falls back to the rule-based
    planner if the optional LLM planner is disabled or fails."""
    if USE_OPENAI_LLM:
        try:
            return llm_tool_plan(
                client,
                transcript=state["messages"][-1]["content"],
                customer_id=state["customer_id"],
                retrieved_memories=state["retrieved_memories"],
            )
        except Exception as exc:
            print("Optional LLM planner failed. Falling back to rule-based planner.")
            print(type(exc).__name__, exc)

    return rule_based_tool_plan(
        transcript=state["messages"][-1]["content"],
        customer_id=state["customer_id"],
        retrieved_memories=state["retrieved_memories"],
    )


# --------------------------------------------------------------------------
# Step 9-10: Confirmation and tool execution
# --------------------------------------------------------------------------

def execute_tool_plan(tools: dict, plan: dict) -> dict:
    tool_name = plan["tool_name"]
    arguments = plan.get("arguments", {})

    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")

    return tools[tool_name](**arguments)


def describe_action_for_confirmation(plan: dict) -> str:
    """Turn a tool plan into the natural spoken question a voice agent would
    actually ask before taking a state-changing action."""
    if plan["tool_name"] == "create_support_ticket":
        priority = plan["arguments"].get("priority", "standard")
        issue_type = plan["arguments"].get("issue_type", "support").replace("_", " ")
        return f"Do you want me to create a {priority}-priority {issue_type} ticket?"
    return f"Do you want me to run {plan['tool_name']}?"


def confirm_and_execute(client: OpenAI, tools: dict, plan: dict, state: dict, auto_confirm: bool = False) -> dict:
    """Ask for human approval before running a state-changing tool call.
    A transcript error should never silently create, send, delete, or update
    anything.

    The confirmation requirement is decided here, by the application, from
    STATE_CHANGING_TOOLS - not from the planner's own "requires_confirmation"
    field. A planner (especially an LLM) can be wrong about that field; the
    application is the layer that actually controls execution.
    """
    print("Proposed action:")
    print(json.dumps(plan, indent=2))

    requires_confirmation = plan["tool_name"] in STATE_CHANGING_TOOLS
    if requires_confirmation != plan.get("requires_confirmation"):
        print(
            "Warning: planner's requires_confirmation "
            f"({plan.get('requires_confirmation')}) disagrees with the "
            f"application's policy ({requires_confirmation}). "
            "Using the application's policy."
        )

    if requires_confirmation:
        state["confirmation"] = "pending"
        question = describe_action_for_confirmation(plan)
        print(f"\n{question}")
        render_spoken_answer(client, question)
        if auto_confirm:
            print("[auto-confirm] Skipping interactive prompt.")
            confirmation = "yes"
        else:
            confirmation = input("Type 'yes' to execute the tool call: ").strip().lower()
        state["confirmation"] = "confirmed" if confirmation == "yes" else "cancelled"
    else:
        confirmation = "yes"

    if confirmation == "yes":
        result = execute_tool_plan(tools, plan)
        print("Tool executed:")
        print(json.dumps(result, indent=2))
        return result

    print("Tool execution cancelled.")
    return {"status": "cancelled", "reason": "User did not confirm the action."}


# --------------------------------------------------------------------------
# Step 11: Response generation
# --------------------------------------------------------------------------

def template_agent_response(
    transcript: str, retrieved_memories: list[dict], tool_plan: dict, tool_result: dict
) -> str:
    top_memory = retrieved_memories[0] if retrieved_memories else None

    if tool_result.get("status") == "created" and "ticket_id" in tool_result:
        return (
            f"I created support ticket {tool_result['ticket_id']} for the password "
            f"reset issue. It is marked as {tool_result['priority']} priority. "
            "The customer should check spam or email filters while support "
            "investigates the missing reset email."
        )

    if tool_result.get("tool") == "search_help_article":
        return f"I found the relevant help guidance. Suggested next step: {tool_result['article']}"

    if tool_result.get("status") == "cancelled":
        return "I did not execute the action because it was not confirmed. No ticket was created."

    if top_memory:
        return (
            f"The request appears related to {top_memory['topic']}. "
            "I found relevant context, but no action was completed."
        )

    return "I could not complete an action, but the request has been captured."


def llm_agent_response(
    client: OpenAI, transcript: str, retrieved_memories: list[dict], tool_plan: dict, tool_result: dict
) -> str:
    """Optional response writer. Only used when USE_OPENAI_LLM=true."""
    retrieved_text = "\n".join(
        f"- {item['id']} / {item['topic']}: {item['text']}" for item in retrieved_memories
    )

    prompt = f"""
You are a support voice agent. Write a concise spoken response.

User transcript:
{transcript}

Retrieved memory:
{retrieved_text}

Tool plan:
{json.dumps(tool_plan, indent=2)}

Tool result:
{json.dumps(tool_result, indent=2)}

Rules:
- Be brief.
- Mention completed actions exactly.
- Do not claim anything that is not supported by the tool result.
- If no action was confirmed, say that no action was taken.
""".strip()

    response = client.responses.create(model=AGENT_MODEL, input=prompt)
    return response.output_text.strip()


def build_final_response(client: OpenAI, state: dict, tool_plan: dict, tool_result: dict) -> str:
    """Report only what actually happened. Falls back to the template
    response if the optional LLM writer is disabled or fails."""
    if USE_OPENAI_LLM:
        try:
            return llm_agent_response(
                client,
                transcript=state["messages"][-1]["content"],
                retrieved_memories=state["retrieved_memories"],
                tool_plan=tool_plan,
                tool_result=tool_result,
            )
        except Exception as exc:
            print("Optional LLM response failed. Falling back to template response.")
            print(type(exc).__name__, exc)

    return template_agent_response(
        transcript=state["messages"][-1]["content"],
        retrieved_memories=state["retrieved_memories"],
        tool_plan=tool_plan,
        tool_result=tool_result,
    )


# --------------------------------------------------------------------------
# Step 12: Text-to-speech
# --------------------------------------------------------------------------

def clean_for_tts(text: str) -> str:
    text = re.sub(r"[*_`#>-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def synthesize_with_openai_tts(client: OpenAI, text: str, output_path: Path) -> Path:
    last_error = None
    for model_name in OPENAI_TTS_MODELS:
        try:
            print(f"Trying OpenAI TTS model: {model_name}")
            with client.audio.speech.with_streaming_response.create(
                model=model_name,
                voice=TTS_VOICE,
                input=text,
                response_format="wav",
            ) as response:
                response.stream_to_file(output_path)
            print("OpenAI TTS succeeded with:", model_name)
            return output_path
        except Exception as exc:
            last_error = exc
            print(f"OpenAI TTS failed for {model_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"No configured OpenAI TTS model worked. Last error: {last_error}")


def synthesize_with_local_tts(text: str, output_path: Path) -> Path | None:
    """Local fallback speech when OpenAI TTS is unavailable for this project."""
    if shutil.which("say"):
        print("Using local macOS 'say' fallback.")
        subprocess.run(["say", "-o", str(TTS_AIFF_OUTPUT), text], check=True)
        if shutil.which("ffmpeg"):
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(TTS_AIFF_OUTPUT), "-ar", "16000", "-ac", "1", str(output_path)],
                check=True,
                capture_output=True,
            )
            return output_path
        return TTS_AIFF_OUTPUT

    if shutil.which("espeak"):
        print("Using local espeak fallback.")
        subprocess.run(["espeak", "-w", str(output_path), text], check=True)
        return output_path

    try:
        import pyttsx3

        print("Using local pyttsx3 fallback.")
        engine = pyttsx3.init()
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()
        return output_path if output_path.exists() else None
    except Exception as exc:
        print("Local pyttsx3 fallback unavailable:", type(exc).__name__, exc)

    return None


def speak_or_print(client: OpenAI, text: str) -> Path | None:
    """Speech output is optional in the architecture: try OpenAI TTS, then
    local speech, then fall back to text only."""
    provider = TTS_PROVIDER if TTS_PROVIDER in {"auto", "openai", "local", "none"} else "auto"
    if provider != TTS_PROVIDER:
        print(f"Unknown TTS_PROVIDER={TTS_PROVIDER!r}. Using auto.")

    if provider in {"auto", "openai"}:
        try:
            return synthesize_with_openai_tts(client, text, TTS_OUTPUT)
        except Exception as exc:
            print("OpenAI TTS unavailable for this project or configuration.")
            print(type(exc).__name__, exc)
            if provider == "openai":
                print("TTS_PROVIDER=openai, so local fallback is disabled.")
                return None

    if provider in {"auto", "local"}:
        try:
            return synthesize_with_local_tts(text, TTS_OUTPUT)
        except Exception as exc:
            print("Local TTS fallback failed.")
            print(type(exc).__name__, exc)
            return None

    return None


def render_spoken_answer(client: OpenAI, final_answer: str) -> None:
    spoken_answer = clean_for_tts(final_answer)
    if len(spoken_answer) > 450:
        spoken_answer = spoken_answer[:450].rstrip() + "."

    output_audio = speak_or_print(client, spoken_answer)

    if output_audio and output_audio.exists():
        print("Saved spoken answer to:", output_audio)
        display(Audio(str(output_audio), autoplay=False))
    else:
        print("No speech output available. Showing text response only.")
        print(spoken_answer)


# --------------------------------------------------------------------------
# Optional production pattern: map the agent loop to LangGraph
# --------------------------------------------------------------------------

def run_langgraph_demo(client: OpenAI, state: dict, tools: dict, auto_confirm: bool) -> dict:
    """Rebuild the same observe -> retrieve -> plan -> confirm -> act -> respond
    loop as an explicit LangGraph state machine. This is an architecture map
    for the production pattern, not a replacement for the manual walkthrough.

    Confirmation uses LangGraph's own human-in-the-loop primitive
    (interrupt() / Command(resume=...)) instead of a bare input() call, and
    the graph is compiled with a checkpointer so its state survives the
    pause. This is what actually implements slide 14's "confirmation prompt
    -> Human-in-the-loop / interrupt" and "logged state -> Checkpoint /
    trace" mapping.
    """
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Command, interrupt
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:
        print("langgraph is not installed. Run: pip install langgraph")
        return {}

    from typing import Literal, TypedDict

    class AgentGraphState(TypedDict, total=False):
        transcript: str
        retrieved_memories: list[dict]
        tool_plan: dict
        confirmed: bool
        tool_result: dict
        final_answer: str

    memory_df = build_memory_frame()

    def node_transcribe(_: AgentGraphState) -> dict:
        return {"transcript": state["messages"][-1]["content"]}

    def node_retrieve_memory(node_state: AgentGraphState) -> dict:
        memories = retrieve_memories(client, memory_df, node_state["transcript"])
        return {"retrieved_memories": memories}

    def node_plan_action(node_state: AgentGraphState) -> dict:
        plan = rule_based_tool_plan(
            transcript=node_state["transcript"],
            customer_id=state["customer_id"],
            retrieved_memories=node_state["retrieved_memories"],
        )
        return {"tool_plan": plan}

    def route_after_plan(node_state: AgentGraphState) -> Literal["confirm_action", "execute_tool", "respond"]:
        # The application decides confirmation from STATE_CHANGING_TOOLS,
        # not the planner's own requires_confirmation field (same policy as
        # confirm_and_execute in the manual path).
        if node_state["tool_plan"]["tool_name"] in STATE_CHANGING_TOOLS:
            return "confirm_action"
        if node_state["tool_plan"].get("tool_name"):
            return "execute_tool"
        return "respond"

    def node_confirm_action(node_state: AgentGraphState) -> dict:
        question = describe_action_for_confirmation(node_state["tool_plan"])
        if auto_confirm:
            print("Proposed action:")
            print(json.dumps(node_state["tool_plan"], indent=2))
            print(f"\n{question}")
            print("[auto-confirm] Skipping interactive prompt.")
            return {"confirmed": True}
        # interrupt() suspends the graph here and persists its state via
        # the checkpointer until resumed with Command(resume=...). On
        # resume, LangGraph re-runs this node from the top, so printing and
        # speaking the question is handled by the driver loop below instead
        # of here - doing it here would print and re-synthesize speech
        # a second time on every resume.
        answer = interrupt({"question": question, "tool_plan": node_state["tool_plan"]})
        return {"confirmed": str(answer).strip().lower() == "yes"}

    def route_after_confirmation(node_state: AgentGraphState) -> Literal["execute_tool", "respond"]:
        return "execute_tool" if node_state.get("confirmed") else "respond"

    def node_execute_tool(node_state: AgentGraphState) -> dict:
        result = execute_tool_plan(tools, node_state["tool_plan"])
        return {"tool_result": result}

    def node_respond(node_state: AgentGraphState) -> dict:
        tool_result = node_state.get(
            "tool_result", {"status": "cancelled", "reason": "User did not confirm the action."}
        )
        answer = template_agent_response(
            transcript=node_state["transcript"],
            retrieved_memories=node_state.get("retrieved_memories", []),
            tool_plan=node_state["tool_plan"],
            tool_result=tool_result,
        )
        return {"tool_result": tool_result, "final_answer": answer}

    builder = StateGraph(AgentGraphState)
    builder.add_node("transcribe", node_transcribe)
    builder.add_node("retrieve_memory", node_retrieve_memory)
    builder.add_node("plan_action", node_plan_action)
    builder.add_node("confirm_action", node_confirm_action)
    builder.add_node("execute_tool", node_execute_tool)
    builder.add_node("respond", node_respond)

    builder.add_edge(START, "transcribe")
    builder.add_edge("transcribe", "retrieve_memory")
    builder.add_edge("retrieve_memory", "plan_action")
    builder.add_conditional_edges("plan_action", route_after_plan)
    builder.add_conditional_edges("confirm_action", route_after_confirmation)
    builder.add_edge("execute_tool", "respond")
    builder.add_edge("respond", END)

    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": state["conversation_id"]}}

    result_state = graph.invoke({}, config=config)

    while "__interrupt__" in result_state:
        pending = result_state["__interrupt__"][0]
        question = pending.value["question"]
        print("Proposed action:")
        print(json.dumps(pending.value["tool_plan"], indent=2))
        print(f"\n{question}")
        render_spoken_answer(client, question)
        answer = input("Type 'yes' to execute the tool call: ").strip().lower()
        result_state = graph.invoke(Command(resume=answer), config=config)

    print("FINAL ANSWER (LangGraph):")
    print(result_state["final_answer"])
    return result_state


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--langgraph",
        action="store_true",
        help="Run the optional LangGraph production-pattern mapping instead of the manual pipeline.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Auto-confirm state-changing tool calls (skip the interactive prompt). Useful for non-interactive runs.",
    )
    args = parser.parse_args()
    pause = not args.yes

    # trust_env=False stops httpx from blocking on macOS system proxy/
    # environment auto-detection before any output is printed.
    client = OpenAI(http_client=httpx.Client(trust_env=False))
    support_tickets: list[dict] = []
    tools = build_tools(support_tickets)

    step_banner("Step 1 - Setup and verification: verify_environment()", pause)
    verify_environment()

    step_banner("Step 2 - Audio preprocessing: prepare_audio()", pause)
    prepared_audio = prepare_audio(INPUT_AUDIO, CLEAN_AUDIO)

    step_banner("Step 3 - Transcription (ASR): transcribe_or_fallback()", pause)
    user_transcript = transcribe_or_fallback(client, prepared_audio)
    print("USER TRANSCRIPT:")
    print(user_transcript)

    step_banner("Step 4 - Conversation state: new_conversation_state()", pause)
    state = new_conversation_state(user_transcript)

    if args.langgraph:
        step_banner("Steps 5-12 - LangGraph production pattern: run_langgraph_demo()", pause)
        # The graph's own retrieve_memory node computes this; skip the
        # duplicate embedding call here.
        run_langgraph_demo(client, state, tools, auto_confirm=args.yes)
        return

    step_banner("Steps 5-6 - Semantic memory and retrieval: retrieve_memories()", pause)
    memory_df = build_memory_frame()
    state["retrieved_memories"] = retrieve_memories(client, memory_df, user_transcript)

    step_banner("Step 8 - Tool plan: build_tool_plan()", pause)
    tool_plan = build_tool_plan(client, state)
    state["pending_action"] = tool_plan
    print("TOOL PLAN:")
    print(json.dumps(tool_plan, indent=2))

    step_banner("Steps 9-10 - Confirmation and tool execution: confirm_and_execute()", pause)
    tool_result = confirm_and_execute(client, tools, tool_plan, state, auto_confirm=args.yes)
    state["tool_results"].append(tool_result)
    state["pending_action"] = None

    step_banner("Step 11 - Response generation: build_final_response()", pause)
    final_answer = build_final_response(client, state, tool_plan, tool_result)
    state["messages"].append({"role": "assistant", "content": final_answer, "timestamp": utc_now()})
    print("FINAL ANSWER:")
    print(final_answer)

    step_banner("Step 12 - Text-to-speech: render_spoken_answer()", pause)
    render_spoken_answer(client, final_answer)

    step_banner("Final - Conversation state", pause)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
