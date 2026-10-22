import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
from harness.metrics import exact_match, token_f1

path = sys.argv[1]
rows = json.load(open(path, encoding="utf-8"))["rows"]
partial = [r for r in rows if exact_match(r["pred"], r["gold"]) == 0
           and max(token_f1(r["pred"], g) for g in r["gold"]) > 0]
print(f"EM=0 but F1>0: {len(partial)} / {len(rows)}")
for r in partial[:12]:
    print("gold:", r["gold"][:2], "| pred:", repr(r["pred"]), "| raw:", repr(r["raw"][:80]))