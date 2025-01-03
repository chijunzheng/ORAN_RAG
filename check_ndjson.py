# check_ndjson.py
import json

with open("part_1.json", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Line {i} is invalid NDJSON: {e}")