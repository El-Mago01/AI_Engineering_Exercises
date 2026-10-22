"""Local tools and validation for the EventOps agent.

Complete the TODO sections as you work through the assignment. The model may
request a tool, but this module validates and executes every request.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
EVENT_ID_PATTERN = re.compile(r"^EVT-\d{3}$")


class ToolContext:
    """Hold file paths and the approval function used by local tools."""

    def __init__(
        self,
        events_path: Path,
        venues_path: Path,
        requests_path: Path,
        approval_callback: Callable[[str], bool],
    ) -> None:
        """Initialize the object with the supplied values."""
        self.events_path = events_path
        self.venues_path = venues_path
        self.requests_path = requests_path
        self.approval_callback = approval_callback


def tool_result(
    ok: bool,
    code: str,
    message: str,
    data: object | None = None,
) -> dict:
    """Create the consistent dictionary returned by every tool."""
    return {"ok": ok, "code": code, "message": message, "data": data}


def prompt_for_approval(summary: str) -> bool:
    """Ask the operator to approve a proposed write operation."""
    print("\nProposed setup request:")
    print(summary)
    answer = input("Create this request? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


DEFAULT_CONTEXT = ToolContext(
    events_path=BASE_DIR / "data" / "events.json",
    venues_path=BASE_DIR / "data" / "venues.json",
    requests_path=BASE_DIR / "data" / "setup_requests.json",
    approval_callback=prompt_for_approval,
)


def load_json(path: Path) -> dict:
    """Load and validate a JSON object from disk."""
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def validate_event_id(event_id: str) -> str:
    """Validate and normalize an EventOps identifier.
    validate_event_id() should:
    require a string;
    remove surrounding spaces;
    convert the value to uppercase;
    match EVT- followed by exactly three digits;
    return the normalized ID as a string.
    " evt-101 " → "EVT-101"
    """
    if not isinstance(event_id, str):
        raise TypeError("event_id must be a string.")

    normalized = event_id.strip().upper()
    if EVENT_ID_PATTERN.fullmatch(normalized):
        return normalized
    raise ValueError("event_id must be a valid Event ID.")


def get_event_details(
    event_id: str,
    context: ToolContext = DEFAULT_CONTEXT,
) -> dict:
    """Return event and venue details for a known event."""
    try:
        normalized = validate_event_id(event_id)
    except (TypeError, ValueError):
        return tool_result(
            ok=False,
            code="invalid_event_id",
            message="The event_id must match EVT- followed by exactly three digits.",
        )

    events = load_json(context.events_path).get("events", {})
    venues = load_json(context.venues_path).get("venues", {})
    event = events.get(normalized)
    if event is None:
        return tool_result(
            ok=False,
            code="event_not_found",
            message=f"Event {normalized} was not found.",
        )

    venue_id = event.get("venue_id")
    if not venue_id:
        return tool_result(
            ok=False,
            code="venue_not_found",
            message=f"Event {normalized} has no assigned venue.",
        )

    venue = venues.get(venue_id)
    if venue is None:
        return tool_result(
            ok=False,
            code="venue_not_found",
            message=f"Venue {venue_id} was not found.",
        )

    return tool_result(
        ok=True,
        code="event_found",
        message="Event details loaded.",
        data={
            "event_id": normalized,
            **event,
            "venue_name": venue.get("name", ""),
            "venue_capacity": venue.get("capacity", ""),
            "available_equipment": venue.get("available_equipment", []),
        },
    )


def check_event_readiness(
    event_id: str,
    context: ToolContext = DEFAULT_CONTEXT,
) -> dict:
    """Check venue capacity and equipment for an event."""
    details_result = get_event_details(event_id, context)
    if not details_result["ok"]:
        return details_result

    details = details_result["data"]
    capacity_ok = details["expected_attendees"] <= details["venue_capacity"]
    available_equipment = set(details["available_equipment"])
    missing_equipment = [
        item
        for item in details["required_equipment"]
        if item not in available_equipment
    ]
    return tool_result(
        ok=True,
        code="readiness_checked",
        message="Event readiness checked.",
        data={
            "event_id": details["event_id"],
            "capacity_ok": capacity_ok,
            "missing_equipment": missing_equipment,
            "ready": capacity_ok and not missing_equipment,
        },
    )


def create_setup_request(
    event_id: str,
    context: ToolContext = DEFAULT_CONTEXT,
) -> dict:
    """Create an approved setup request for an event needing attention.
    create_setup_request writes to data/setup_requests.json. Before writing,
    it must calculate the current readiness result and show the exact issues to the operator."""
    readiness = check_event_readiness(event_id, context)
    if not readiness["ok"]:
        return readiness

    if readiness["data"]["ready"]:
        return tool_result(
            ok=False,
            code="event_already_ready",
            message="No setup request is needed.",
            data=readiness["data"],
        )

    readiness_data = readiness["data"]
    requests = load_json(context.requests_path)
    if not isinstance(requests.get("requests"), list):
        return tool_result(
            ok=False,
            code="storage_error",
            message="The setup request file is invalid.",
            data=None,
        )

    existing_open_request = next(
        (
            request
            for request in requests["requests"]
            if isinstance(request, dict)
            and request.get("event_id") == readiness_data["event_id"]
            and request.get("status") == "open"
        ),
        None,
    )
    if existing_open_request is not None:
        return tool_result(
            ok=False,
            code="duplicate_open_request",
            message="An open setup request already exists for this event.",
            data=existing_open_request,
        )

    details_result = get_event_details(event_id, context)
    if not details_result["ok"]:
        return details_result

    details = details_result["data"]
    issues = []
    if not readiness_data["capacity_ok"]:
        issues.append(
            "capacity too low: "
            f"{details['expected_attendees']} expected attendees; "
            f"venue capacity {details['venue_capacity']}"
        )
    if readiness_data["missing_equipment"]:
        issues.append(
            "missing equipment: "
            + ", ".join(readiness_data["missing_equipment"])
        )

    summary = f"{readiness_data['event_id']}: " + "; ".join(issues)

    if not context.approval_callback(summary):
        return tool_result(
            ok=False,
            code="approval_rejected",
            message="The setup request was not created.",
            data=None,
        )

    request = {
        "request_id": f"REQ-{len(requests["requests"]) + 1:03d}",
        "event_id": readiness_data["event_id"],
        "issues": summary,
        "status": "open",
    }
    requests["requests"].append(request)
    context.requests_path.write_text(
        json.dumps(requests, indent=2) + "\n",
        encoding="utf-8",
    )
    return tool_result(
        ok=True,
        code="setup_request_created",
        message="The setup request was created.",
        data={
            "request_id": request["request_id"],
            "event_id": readiness_data["event_id"],
            "issues": summary,
            "status": "open",
        },
    )


TOOL_REGISTRY = {
    "get_event_details": {
        "handler": get_event_details,
        "requires_approval": False,
    },
    "check_event_readiness": {
        "handler": check_event_readiness,
        "requires_approval": False,
    },
    "create_setup_request": {
        "handler": create_setup_request,
        "requires_approval": True,
    },
}


def validate_call(name: str, arguments: dict) -> dict:
    """Validate a model-proposed tool name and argument object."""
    # Checks the TOOL_REGISTRY and require exactly one event_id argument.
    if not isinstance(name, str) or name not in TOOL_REGISTRY:
        return tool_result(
            False,
            "unknown_tool",
            "The requested tool is not allowed.",
        )
    if not isinstance(arguments, dict):
        return tool_result(
            False,
            "invalid_arguments",
            "Tool arguments must be an object.",
        )
    if set(arguments) != {"event_id"}:
        return tool_result(
            False,
            "invalid_arguments",
            "The tool requires exactly one event_id argument.",
        )
    try:
        event_id = validate_event_id(arguments["event_id"])
    except (TypeError, ValueError):
        return tool_result(
            False,
            "invalid_event_id",
            "The event_id must match EVT- followed by exactly three digits.",
        )
    return tool_result(
        ok=True,
        code="call_validated",
        message="The tool call is valid.",
        data={"event_id": event_id},
    )


def dispatch(
    name: str,
    arguments: dict,
    context: ToolContext = DEFAULT_CONTEXT,
) -> dict:
    """Validate and execute one allowed and listed tool call."""
    validated_call = validate_call(name, arguments)
    if not validated_call["ok"]:
        return validated_call
    spec = TOOL_REGISTRY[name]

    try:
        return spec["handler"](
            validated_call["data"]["event_id"],
            context,
        )
    except Exception:
        return tool_result(
            ok=False,
            code="tool_error",
            message="The tool could not be executed.",
        )


def main():
    get_event_details("evt-101")


if __name__ == "__main__":
    main()
