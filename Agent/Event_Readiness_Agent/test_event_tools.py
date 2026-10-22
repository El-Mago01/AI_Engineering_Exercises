"""Deterministic tests for the local EventOps tools."""

import json
import tempfile
import unittest
from pathlib import Path

from event_tools import ToolContext, create_setup_request


class CreateSetupRequestTests(unittest.TestCase):
    """Verify setup-request creation safeguards."""

    def test_existing_open_request_prevents_duplicate_without_approval(self) -> None:
        """An open request for the normalized event ID must block a duplicate."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            events_path = directory / "events.json"
            venues_path = directory / "venues.json"
            requests_path = directory / "setup_requests.json"

            events_path.write_text(
                json.dumps(
                    {
                        "events": {
                            "EVT-102": {
                                "name": "Design Systems Workshop",
                                "date": "2026-08-15",
                                "expected_attendees": 32,
                                "venue_id": "VEN-B",
                                "required_equipment": [
                                    "projector",
                                    "wifi",
                                    "wireless microphone",
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            venues_path.write_text(
                json.dumps(
                    {
                        "venues": {
                            "VEN-B": {
                                "name": "Studio Two",
                                "capacity": 40,
                                "available_equipment": ["projector", "wifi"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            request_document = {
                "requests": [
                    {
                        "request_id": "REQ-001",
                        "event_id": "EVT-102",
                        "issues": "missing equipment: wireless microphone",
                        "status": "open",
                    }
                ]
            }
            requests_path.write_text(
                json.dumps(request_document, indent=2) + "\n",
                encoding="utf-8",
            )
            original_contents = requests_path.read_text(encoding="utf-8")
            approval_summaries = []
            context = ToolContext(
                events_path=events_path,
                venues_path=venues_path,
                requests_path=requests_path,
                approval_callback=lambda summary: approval_summaries.append(summary) or True,
            )

            result = create_setup_request(" evt-102 ", context)

            self.assertFalse(result["ok"])
            self.assertEqual(result["code"], "duplicate_open_request")
            self.assertEqual(result["data"], request_document["requests"][0])
            self.assertEqual(approval_summaries, [])
            self.assertEqual(
                requests_path.read_text(encoding="utf-8"),
                original_contents,
            )


if __name__ == "__main__":
    unittest.main()
