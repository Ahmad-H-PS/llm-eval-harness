import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8")]

print("n:", len(rows))
print("pred counts:", dict(Counter(r["pred"] for r in rows)))
print("gold counts:", dict(Counter(r["gold"] for r in rows)))
print("accuracy (unparseable = wrong):", sum(r["pred"] == r["gold"] for r in rows) / len(rows))

print("\nfirst 5 raw outputs, then up to 5 unparseable ones:")
for r in rows[:5] + [r for r in rows if r["pred"] is None][:5]:
    print(r["gold"], r["pred"], repr(r["raw"]))