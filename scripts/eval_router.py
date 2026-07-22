#!/usr/bin/env python3
"""Neural Gateway Parser Accuracy Evaluator.
Tests the FULL parse_task_request() pipeline, not just the embedding parser in isolation.
Run from neural_gateway/: python scripts/eval_router.py
"""
import json, sys, io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Use parse_task_request (full pipeline) NOT parse_prompt_to_semantic_struct (embedding only)
from app.core.semantic_parser import parse_task_request

def main():
    golden = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'golden_eval.json').read_text(encoding='utf-8'))
    fam_hits = risk_hits = dom_hits = 0
    total = len(golden)
    failures = []
    for case in golden:
        try:
            task = parse_task_request(prompt=case['text'])
        except Exception as e:
            print(f"ERROR parsing '{case['text'][:50]}': {e}")
            continue
        acceptable_fam = case.get('acceptable_families', [case['primary_family']])
        acceptable_dom = case.get('acceptable_domains', [case['domain']])
        acceptable_risk = case.get('acceptable_risks', [case['risk_tier']])
        fam_ok = task.primary_family in acceptable_fam
        risk_ok = task.risk_tier in acceptable_risk
        dom_ok = task.domain in acceptable_dom
        if fam_ok: fam_hits += 1
        if risk_ok: risk_hits += 1
        if dom_ok: dom_hits += 1
        if not (fam_ok and risk_ok and dom_ok):
            failures.append({
                'prompt': case['text'][:80],
                'exp_fam': acceptable_fam, 'got_fam': task.primary_family,
                'exp_risk': acceptable_risk, 'got_risk': task.risk_tier,
                'exp_dom': acceptable_dom, 'got_dom': task.domain,
                'fam_ok': fam_ok, 'risk_ok': risk_ok, 'dom_ok': dom_ok
            })
    fam_acc = fam_hits / total
    risk_acc = risk_hits / total
    dom_acc = dom_hits / total
    print(f'\n{"="*60}')
    print('NEURAL GATEWAY PARSER ACCURACY REPORT')
    print(f'{"="*60}')
    print(f'Total: {total}')
    print(f'Family Accuracy: {fam_hits}/{total} = {fam_acc:.1%}')
    print(f'Risk Accuracy:   {risk_hits}/{total} = {risk_acc:.1%}')
    print(f'Domain Accuracy: {dom_hits}/{total} = {dom_acc:.1%}')
    if failures:
        print(f'\nFailed ({len(failures)} cases):')
        for f in failures:
            tags = []
            if not f['fam_ok']: tags.append(f"FAM: {f['exp_fam']} -> {f['got_fam']}")
            if not f['risk_ok']: tags.append(f"RISK: {f['exp_risk']} -> {f['got_risk']}")
            if not f['dom_ok']: tags.append(f"DOM: {f['exp_dom']} -> {f['got_dom']}")
            print(f"  FAIL [{', '.join(tags)}]")
            print(f"       Prompt: '{f['prompt']}'")
    THRESHOLD = 0.80
    passed = fam_acc >= THRESHOLD and risk_acc >= THRESHOLD and dom_acc >= THRESHOLD
    print(f'\n{"PASS" if passed else "FAIL"} (threshold {THRESHOLD:.0%})')
    sys.exit(0 if passed else 1)

if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    main()
