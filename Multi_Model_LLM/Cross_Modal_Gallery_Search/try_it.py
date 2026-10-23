"""Run one gallery search and print a compact result summary."""

import json
import subprocess
import sys


command = [sys.executable, "gallery_search.py", "--query", "a sleeping pet", "--top-k", "2"]
completed = subprocess.run(command, capture_output=True, text=True)
if completed.returncode:
    detail = completed.stderr.strip().splitlines()[-1]
    print(f"SEARCH FAIL: {detail[:160]}")
    raise SystemExit(1)
results = json.loads(completed.stdout)
print(f"Matches: {len(results)}")
for position, result in enumerate(results[:2], start=1):
    print(f"{position}. {result['source']} ({result['score']:.3f})")
