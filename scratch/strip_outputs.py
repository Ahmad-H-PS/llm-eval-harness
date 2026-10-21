import json
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "notebooks/01_prototype.ipynb"
with open(p, encoding="utf-8") as f:
    nb = json.load(f)

for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    kept = []
    for o in c["outputs"]:
        text = "".join(o.get("text", []))
        plain = "".join(o.get("data", {}).get("text/plain", []))
        if o.get("output_type") == "stream" and o.get("name") == "stderr" and "Users" in text:
            continue  # local path in the tqdm warning
        if o.get("output_type") == "execute_result" and "Qwen2ForCausalLM" in plain:
            continue  # architecture dump
        kept.append(o)
    c["outputs"] = kept

with open(p, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")