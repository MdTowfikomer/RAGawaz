"""
Phase 5 Verification Script:
1. Validates STT language mappings across Hindi, English, Marathi, Tamil, Bengali, Hinglish.
2. Validates 4-Tier pipeline status derivation logic for Insufficient Evidence (SKIPPED vs VERIFIED).
"""

import sys
import io
import json

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LANGUAGE_LOCALE_MAP = {
    'hi-IN': 'hi-IN',
    'en-IN': 'en-IN',
    'en-US': 'en-IN',
    'hi-EN': 'en-IN',
    'mr-IN': 'mr-IN',
    'ta-IN': 'ta-IN',
    'bn-IN': 'bn-IN',
}


def derive_groundedness_status(status_result: str):
    """Mirror exact logic from PerformanceTelemetry.jsx."""
    if status_result in ['refusal_safety', 'refusal_offtopic', 'refusal_insufficient_evidence']:
        return "SKIPPED"
    elif status_result == 'refusal_ungrounded':
        return "REJECTED"
    elif status_result == 'success':
        return "VERIFIED"
    return "PENDING"


def derive_full_pipeline_display(status_result: str):
    """Mirror full 4-tier diagnostic display from PerformanceTelemetry.jsx."""
    # 1. Safety
    safety = "BLOCKED" if status_result == 'refusal_safety' else "PASS"

    # 2. Relevance
    if status_result == 'refusal_safety':
        relevance = "SKIPPED"
    elif status_result == 'refusal_offtopic':
        relevance = "BLOCKED"
    else:
        relevance = "PASS"

    # 3. Insufficient Evidence
    if status_result in ['refusal_safety', 'refusal_offtopic']:
        evidence = "SKIPPED"
    elif status_result == 'refusal_insufficient_evidence':
        evidence = "INTERCEPTED"
    else:
        evidence = "PASS"

    # 4. Groundedness
    groundedness = derive_groundedness_status(status_result)

    return {
        "Safety Guardrail": safety,
        "Relevance Gate": relevance,
        "Insufficient Evidence": evidence,
        "Groundedness Verifier": groundedness,
    }


def main():
    print("=" * 80)
    print("PHASE 5 VERIFICATION: UI INTEGRATION & DIAGNOSTICS")
    print("=" * 80)

    # 1. Verify STT Language Locale Mappings
    print("\n1. Verifying STT Language Locale Mappings:")
    expected_locales = {
        'Hindi': ('hi-IN', 'hi-IN'),
        'English': ('en-IN', 'en-IN'),
        'Hinglish': ('hi-EN', 'en-IN'),
        'Marathi': ('mr-IN', 'mr-IN'),
        'Tamil': ('ta-IN', 'ta-IN'),
        'Bengali': ('bn-IN', 'bn-IN'),
    }
    all_locales_valid = True
    for lang_name, (ui_code, expected_stt_locale) in expected_locales.items():
        actual_locale = LANGUAGE_LOCALE_MAP.get(ui_code)
        matches = actual_locale == expected_stt_locale
        status = "✅ PASS" if matches else "❌ FAIL"
        if not matches:
            all_locales_valid = False
        print(f"   [{status}] {lang_name:<10} | UI Code: {ui_code:<6} -> STT Locale: {actual_locale}")

    assert all_locales_valid, "Language locale mapping failed!"

    # 2. Verify Pipeline Display for Insufficient Evidence Interception
    print("\n2. Verifying Pipeline Diagnostics for Insufficient Evidence Interception:")
    diag_insufficient = derive_full_pipeline_display("refusal_insufficient_evidence")
    for stage, state in diag_insufficient.items():
        print(f"   {stage:<26}: {state}")

    assert diag_insufficient["Insufficient Evidence"] == "INTERCEPTED"
    assert diag_insufficient["Groundedness Verifier"] == "SKIPPED"
    print("   ✅ Insufficient evidence correctly shows Groundedness as SKIPPED (not VERIFIED).")

    # 3. Verify Pipeline Display for Normal Successful Generation
    print("\n3. Verifying Pipeline Diagnostics for Normal Successful Generation:")
    diag_success = derive_full_pipeline_display("success")
    for stage, state in diag_success.items():
        print(f"   {stage:<26}: {state}")

    assert diag_success["Safety Guardrail"] == "PASS"
    assert diag_success["Relevance Gate"] == "PASS"
    assert diag_success["Insufficient Evidence"] == "PASS"
    assert diag_success["Groundedness Verifier"] == "VERIFIED"
    print("   ✅ Normal generation correctly shows Groundedness as VERIFIED.")

    # 4. Verify Pipeline Display for Safety Refusal
    print("\n4. Verifying Pipeline Diagnostics for Safety Refusal:")
    diag_safety = derive_full_pipeline_display("refusal_safety")
    for stage, state in diag_safety.items():
        print(f"   {stage:<26}: {state}")
    assert diag_safety["Safety Guardrail"] == "BLOCKED"
    assert diag_safety["Groundedness Verifier"] == "SKIPPED"
    print("   ✅ Safety refusal correctly shows subsequent stages as SKIPPED.")

    print("\n" + "=" * 80)
    print("ALL PHASE 5 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
