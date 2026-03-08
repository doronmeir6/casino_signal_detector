"""Offline OpenAI extraction: conversation -> canonical JSON schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

# Allow running as `python src/extract_with_llm.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.signal_schema import (
    SIGNAL_CATEGORIES,
    extraction_to_labels,
    validate_and_normalize_extraction,
)

load_dotenv()

SYSTEM_PROMPT = """You extract casino guest signals from conversation text.
Return strict JSON only. No markdown.
Signals are: intent, value, sentiment, life_event, competitive.
For INTENT:
- detected=true only when the guest explicitly expresses a plan/request to book, visit, reserve, or schedule.
- detected=false for purely administrative/info turns (e.g., asking only for address/postcode/reference number).
Each signal must include:
- detected (bool)
- confidence (0..1)
- evidence (short verbatim phrase from guest text if detected else "")
Also return:
- guest_text (string, concatenated guest turns)
- parser_confidence (0..1 overall confidence)
- notes (short string)
"""

_INTENT_POSITIVE_HINTS = [
    "book", "booking", "reserve", "reservation", "plan", "planning", "trip",
    "visit", "coming", "come in", "stay", "schedule", "table for", "room for",
]
_INTENT_ADMIN_ONLY_HINTS = [
    "reference number", "postcode", "address", "phone number", "contact number",
    "confirm my reservation number", "reservation number",
]


def _conversation_hash(conversation: list[dict]) -> str:
    payload = json.dumps(conversation, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_guest_text(conversation: list[dict]) -> str:
    return " ".join(
        m.get("content", "")
        for m in conversation
        if m.get("role") == "guest"
    ).strip()


def _build_user_prompt(record: dict) -> str:
    conversation = record.get("conversation", [])
    return (
        "Conversation turns:\n"
        f"{json.dumps(conversation, ensure_ascii=False)}\n\n"
        "Task:\n"
        "1) Build guest_text from guest turns only.\n"
        "2) Extract each signal with detected/confidence/evidence.\n"
        "3) Use semantic meaning, not only keywords.\n"
        "4) Return strict JSON object with fields:\n"
        '{'
        '"guest_text": "...", '
        '"signals": {'
        '"intent":{"detected":bool,"confidence":0-1,"evidence":"..."},'
        '"value":{"detected":bool,"confidence":0-1,"evidence":"..."},'
        '"sentiment":{"detected":bool,"confidence":0-1,"evidence":"..."},'
        '"life_event":{"detected":bool,"confidence":0-1,"evidence":"..."},'
        '"competitive":{"detected":bool,"confidence":0-1,"evidence":"..."}'
        "}, "
        '"parser_confidence": 0-1, '
        '"notes":"..."'
        "}"
    )


def _postprocess_intent_signal(extraction: dict) -> dict:
    """Reduce intent false positives from admin-only requests."""
    guest_text = str(extraction.get("guest_text", "") or "").lower()
    sig = extraction.get("signals", {}).get("intent", {})
    if not isinstance(sig, dict):
        return extraction

    has_positive_cue = any(k in guest_text for k in _INTENT_POSITIVE_HINTS)
    has_admin_only_cue = any(k in guest_text for k in _INTENT_ADMIN_ONLY_HINTS)

    # Downgrade admin-only intent detections when no booking/visit cue exists.
    if bool(sig.get("detected", False)) and (not has_positive_cue) and has_admin_only_cue:
        sig["detected"] = False
        sig["confidence"] = min(float(sig.get("confidence", 0.0) or 0.0), 0.35)
        sig["evidence"] = ""
        extraction.setdefault("notes", "")
        extraction["notes"] = (str(extraction["notes"]) + " | intent postprocess: admin-only downgraded").strip(" |")

    extraction.setdefault("signals", {})["intent"] = sig
    return extraction


def _call_llm_with_retries(client, model: str, messages: list[dict], retries: int = 3) -> Optional[str]:
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=800,
            )
            return resp.choices[0].message.content
        except Exception:
            if attempt == retries:
                return None
            time.sleep(1.5 * attempt)
    return None


def _load_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    cache: dict[str, dict] = {}
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = row.get("hash")
                extraction = row.get("extraction")
                if key and isinstance(extraction, dict):
                    cache[key] = extraction
            except json.JSONDecodeError:
                continue
    return cache


def _append_cache(cache_path: Path, key: str, extraction: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"hash": key, "extraction": extraction}, ensure_ascii=False) + "\n")


def run_extraction(
    input_path: str,
    output_jsonl: str,
    cache_path: str,
    report_path: str,
    model: str = "gpt-4o-mini",
    min_parser_confidence: float = 0.45,
    retries: int = 3,
) -> None:
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found. Add it to .env or env vars.")
    client = openai.OpenAI(api_key=api_key)

    in_path = Path(input_path)
    out_path = Path(output_jsonl)
    cache_file = Path(cache_path)
    report_file = Path(report_path)

    with open(in_path, encoding="utf-8") as f:
        records = json.load(f)

    cache = _load_cache(cache_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": len(records),
        "from_cache": 0,
        "llm_calls": 0,
        "ok": 0,
        "schema_failures": 0,
        "llm_failures": 0,
        "low_confidence_filtered": 0,
    }
    failures: list[dict] = []

    with open(out_path, "w", encoding="utf-8") as out:
        for rec in tqdm(records, desc="LLM extraction", unit="conv"):
            conv = rec.get("conversation", [])
            conv_hash = _conversation_hash(conv)
            conv_id = rec.get("id")
            source = rec.get("source", "unknown")

            extraction = None
            if conv_hash in cache:
                extraction = cache[conv_hash]
                stats["from_cache"] += 1
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(rec)},
                ]
                content = _call_llm_with_retries(
                    client,
                    model=model,
                    messages=messages,
                    retries=retries,
                )
                stats["llm_calls"] += 1
                if content is None:
                    stats["llm_failures"] += 1
                    failures.append({"id": conv_id, "reason": "llm_call_failed"})
                    continue
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    stats["schema_failures"] += 1
                    failures.append({"id": conv_id, "reason": "invalid_json"})
                    continue
                extraction = payload

            # Canonical normalization
            is_valid, normalized, errors = validate_and_normalize_extraction(extraction)
            normalized["conversation_id"] = conv_id
            if not normalized.get("guest_text"):
                normalized["guest_text"] = _extract_guest_text(conv)

            if not is_valid:
                stats["schema_failures"] += 1
                failures.append({"id": conv_id, "reason": "schema_validation", "errors": errors[:5]})
                continue

            normalized = _postprocess_intent_signal(normalized)

            if float(normalized.get("parser_confidence", 0.0)) < min_parser_confidence:
                stats["low_confidence_filtered"] += 1
                failures.append({"id": conv_id, "reason": "low_parser_confidence"})
                continue

            labels = extraction_to_labels(normalized)
            record_out = {
                "id": conv_id,
                "source": source,
                "conversation": conv,
                "extraction": normalized,
                "labels": labels,
                "original_labels": rec.get("labels", {}),
                "cache_key": conv_hash,
            }

            out.write(json.dumps(record_out, ensure_ascii=False) + "\n")
            _append_cache(cache_file, conv_hash, normalized)
            stats["ok"] += 1

    # Aggregate label counts for report
    label_counts = {k: 0 for k in SIGNAL_CATEGORIES}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for cat, v in row.get("labels", {}).items():
                    if cat in label_counts:
                        label_counts[cat] += int(v)

    report = {
        "input_path": str(in_path),
        "output_jsonl": str(out_path),
        "cache_path": str(cache_file),
        "model": model,
        "provider": "openai",
        "min_parser_confidence": min_parser_confidence,
        "stats": stats,
        "label_counts": label_counts,
        "failure_examples": failures[:80],
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved extracted rows -> {out_path}")
    print(f"Saved extraction report -> {report_file}")
    print(f"Rows kept: {stats['ok']} / {stats['total']}")
    print(f"Label counts: {label_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM extraction pipeline for training data")
    parser.add_argument("--input", default="data/raw/conversations_merged.json")
    parser.add_argument("--output_jsonl", default="data/processed/extractions.jsonl")
    parser.add_argument("--cache_path", default="data/processed/extraction_cache.jsonl")
    parser.add_argument("--report_path", default="data/processed/extraction_report.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--min_parser_confidence", type=float, default=0.45)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    run_extraction(
        input_path=args.input,
        output_jsonl=args.output_jsonl,
        cache_path=args.cache_path,
        report_path=args.report_path,
        model=args.model,
        min_parser_confidence=args.min_parser_confidence,
        retries=args.retries,
    )
