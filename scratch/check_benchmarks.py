import json
with open('app/models/real_benchmarks.json') as f:
    data = json.load(f)
for m in data:
    if "qwen" in m["model"].lower() or "gemini-2.0" in m["model"].lower() or "claude" in m["model"].lower() or "llama" in m["model"].lower():
        print(f"{m['model']}: Coding={m.get('coding', 0)}")
