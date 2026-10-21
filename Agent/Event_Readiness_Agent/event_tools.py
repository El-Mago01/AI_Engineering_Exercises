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
    # : Validates the ID, load both data files, and return a result dictionary.
    events = load_json(context.events_path).get("events", {})
    venues = load_json(context.venues_path).get("venues", {})
    try:
        event_id = validate_event_id(event_id)
    except ValueError as e:
        tool_result(
            ok=False,
            code="invalid_event_id",
            message="Provided event_id invalid",
            data=None
        )

    if event_id not in events.keys():
        return tool_result(
            ok=False,
            code="event_id_not_available",
            message="Provided event_id is not present in DB",
            data=None
        )
    event_details = events[event_id]
    venue_id = event_details.get("venue_id","")
    if venue_id == "":
        return tool_result(
            ok=False,
            code="venue_id_not_found_in_event_db",
            message="No venue_id not stored in event.json",
            data=None
        )
    venue_details = venues.get(venue_id,{})
    if venue_details == {}:
        return tool_result(
            ok=False,
            code="venue_id_not_found_in_venues_db",
            message="venue_id not found in venues.json",
            data=None
        )
    return tool_result(
            ok=True,
            code="event_details_returned",
            message="All event details returned",
            data={
                "event_id": event_id,
                **event_details,
                "venue_name": venue_details.get("name",""),
                "venue_capacity": venue_details.get("capacity",""),
                "available_equipment": venue_details.get("available_equipment","")
            }
        )


def check_event_readiness(
    event_id: str,
    context: ToolContext = DEFAULT_CONTEXT,
) -> dict:
    """Check venue capacity and equipment for an event."""
    # Compares attendance and equipment with the assigned venue.
    event_details = get_event_details(event_id, context)
    if not event_details.get("ok",""):
        return tool_result(
            ok=False,
            code="event_details_not_found",
            message="Could not get event details",
            data={
                "event_id": event_id,
                "capacity_ok": False,
                "missing_equipment": [],
                "ready": False
            }
        )
    if not event_details.get("data","").get("available_equipment",""):
        return tool_result(
            ok=False,
            code="missing_equipment",
            message="available_equipment field not present",
            data={
                "event_id": event_id,
                "capacity_ok": False,
                "missing_equipment": [],
                "ready": False
            }
        )
        return event_details
    if not event_details.get("data").get("expected_attendees",""):
        return tool_result(
            ok=False,
            code="expected_attendees_field_missing_in_returned_data",
            message="expected_attendees field not present",
            data={
                "event_id": event_id,
                "capacity_ok": False,
                "missing_equipment": [],
                "ready": False
            }
        )

    capacity_ok = (
            event_details["data"]["expected_attendees"] <= event_details["data"]["venue_capacity"])
    available_equipment = set(event_details["data"]["available_equipment"])
    missing_equipment = [
        item
        for item in event_details["data"]["required_equipment"]
        if item not in available_equipment
    ]
    return tool_result(
        ok=capacity_ok and not missing_equipment,
        code="result_of_readiness_check",
        message="result of readiness check",
        data={
            "event_id": event_id,
            "capacity_ok": capacity_ok,
            "missing_equipment": missing_equipment,
            "ready": capacity_ok and not missing_equipment
        }
    )


def create_setup_request(
    event_id: str,
    context: ToolContext = DEFAULT_CONTEXT,
) -> dict:
    """Create an approved setup request for an event needing attention.
    create_setup_request writes to data/setup_requests.json. Before writing,
    it must calculate the current readiness result and show the exact issues to the operator."""
    readiness = check_event_readiness(event_id, context)
    if readiness["data"]["ready"]:
        tool_result(
            ok=True,
            code="readiness_ok_no_setup_needed",
            message="No setup request is needed",
            data={
                **readiness["data"],
            }
        )
    capacity_ok = readiness["data"]['capacity_ok']
    missing_equipment = readiness["data"]['missing_equipment']
    if capacity_ok:
        capacity_issue = "No capacity issue"
    else:
        if "expected_attendees" not in readiness["data"] or "venue_capacity" not in readiness["data"]:
            capacity_issue = "There is a capacity issue, exact numbers are missing"
        else:
            expected_attendees = readiness["data"]["expected_attendees"]
            venue_capacity = readiness["data"]["venue_capacity"]
            capacity_issue = f"Capacity issue: There are {expected_attendees} attendees and the venue capacity is for {venue_capacity} attendees"
    if not missing_equipment:
        missing_equipment_issue = "No equipment issue"
    else:
        missing_equipment_issue = f"Equipment missing issue: The following equipments are missing: {missing_equipment}"

    summary=f"{readiness["data"]['event_id']}: {missing_equipment_issue}, {capacity_issue} "

    if not context.approval_callback(summary):
        return tool_result(
            ok=False,
            code="approval_rejected",
            message="The setup request was not created.",
            data={}
        )
    requests = load_json(context.requests_path)
    if not "requests" in requests.keys() or not isinstance(requests["requests"], list):
        return tool_result(
            ok=False,
            code="storage_error",
            message="The json file for requests is corrupted",
            data={}
        )
    request = {
        "request_id": f"REQ-{len(requests["requests"]) + 1:03d}",
        "event_id": readiness["data"]["event_id"],
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
        code="result_of_readiness_check",
        message="result of readiness check",
        data={
            "event_id": event_id,
            "capacity_ok": capacity_ok,
            "request_id" : request["request_id"],
            "missing_equipment": missing_equipment,
            "ready": False
        }
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
            "unknown_tool",
            "The requested tool is not allowed.",
        )
    if set(arguments) != {"event_id"}:
        return tool_result(
            False,
            "invalid_arguments",
            "The tool requires exactly one event_id argument.",
        )
    try:
        event_id = validate_event_id(arguments["event_id"])
    except ValueError:
        return tool_result(
            False,
            "unknown_tool",
            "The requested tool has invalid arguments (invalid event_id).",
        )
    return tool_result(
        ok=True,
        code="OK",
        message="The tool is validated",
        data={"event_id": event_id,
              "ready" : True},
    )

def dispatch(
    name: str,
    arguments: dict,
    context: ToolContext = DEFAULT_CONTEXT,
) -> dict:
    """Validate and execute one allowed and listed tool call.
    The dispatcher is the single entry point between a model-proposed call and a local Python function.
    """
    validated_call = validate_call(name, arguments)
    if not validated_call.get("ok", False) or not validated_call.get("data", {}):
        return validated_call
    spec = TOOL_REGISTRY[name]

    return spec["handler"](
        validated_call["data"]["event_id"],
        context,
)

def main():
    get_event_details("evt-101")

if __name__ == "__main__":
    main()