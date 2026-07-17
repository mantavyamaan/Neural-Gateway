import json
import sys
import os

sys.path.append(r"C:\Users\manta\OneDrive\Desktop\neural_gateway")
from app.core.embedding_parser import get_parser

parser = get_parser()
eval_file = r"C:\Users\manta\OneDrive\Desktop\neural_gateway\data\golden_eval.json"
with open(eval_file, "r", encoding="utf-8") as f:
    prompts = json.load(f)

for p in prompts[:10]:
    res = parser.parse(p['text'])
    print(f"TEXT: {p['text'][:50]}")
    print(f"  Expected: fam={p['primary_family']} dom={p['domain']} risk={p['risk_tier']}")
    print(f"  Actual:   fam={res.primary_family} dom={res.domain} risk={res.risk_tier}")
    match = res.primary_family == p['primary_family'] and res.domain == p['domain'] and res.risk_tier == p['risk_tier']
    print(f"  Match:    {match}\n")
