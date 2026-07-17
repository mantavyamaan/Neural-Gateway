import sys
sys.path.insert(0, ".")
from app.core.database import get_all_models
models = get_all_models()
for m in models:
    if "qwen" in m["name"].lower() or "gemini" in m["name"].lower() or "claude" in m["name"].lower():
        p = m.get('performance', {})
        c = p.get('coding', 0)
        print(f"{m['name']}: Coding={c:.3f}")
