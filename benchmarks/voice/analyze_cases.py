import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

with open("benchmarks/voice/case_inspection_results.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

print(f"Total cases: {len(cases)}")

by_lang = {}
for c in cases:
    l = c["lang"]
    if l not in by_lang:
        by_lang[l] = []
    by_lang[l].append(c)

for l, lcases in by_lang.items():
    print(f"\n=================== LANGUAGE: {l.upper()} ===================")
    for c in lcases:
        hit = c["retrieved_hit"]
        status = c["status"]
        has_key = c["has_gold_key"]
        v_score = c["verifier_score"]
        v_method = c["verifier_method"]
        
        # Categorize
        cat = "UNKNOWN"
        if status == "success":
            if has_key and hit:
                cat = "TRUE_CORRECT"
            elif has_key and not hit:
                cat = "DISTRACTOR_HIT"
            elif not has_key and hit:
                cat = "EVAL_KEYWORD_ARTIFACT"
            elif not has_key and not hit:
                cat = "VERIFIER_FALSE_POSITIVE"
        else:
            if hit:
                cat = "FALSE_REFUSAL"
            else:
                cat = "TRUE_REFUSAL"
                
        print(f"[{cat}] Concept: {c['concept']}")
        print(f"  Query: {c['query']}")
        print(f"  Hit: {hit} (Top Score: {c['retrieved_top_score']:.3f}) | Status: {status}")
        print(f"  Verifier: {c['verifier_is_grounded']} (method: {v_method}, score: {v_score:.3f})")
        print(f"  Answer: {c['answer'][:90]}...")
        if not hit and status == "success":
            print(f"  Retrieved Top Text: {c['retrieved_top_text'][:90]}...")
        print()
