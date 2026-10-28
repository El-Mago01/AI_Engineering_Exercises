"""Run the small deterministic team before trying the live API example."""

from agent_team import run_team


result = run_team()
print(result["brief"])
print(f"Specialists completed: {len(result['results'])}")
print(f"Status: {result['status']}")
