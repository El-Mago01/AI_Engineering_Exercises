"""Try one local EventOps tool dispatch."""

from event_tools import dispatch

#
# print("Valid tool call:")
# print(dispatch("get_event_details", {"event_id": "evt-101"}))
#
# print("\nUnknown tool call:")
# print(dispatch("delete_event", {"event_id": "EVT-101"}))

print("\nStore a request:")
print(dispatch("create_setup_request", {"event_id": "EVT-999"}))