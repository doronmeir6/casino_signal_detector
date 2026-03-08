"""
Canonical schema for LLM-extracted casino signals.

This schema is shared by:
  - Offline extraction (`extract_with_llm.py`)
  - Online inference extraction (`infer_extract_llm.py`)
  - Table flattening / feature generation
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

SIGNAL_CATEGORIES = ["intent", "value", "sentiment", "life_event", "competitive"]


def default_signal_block() -> dict[str, dict[str, Any]]:
    return {
        cat: {
            "detected": False,
            "confidence": 0.0,
            "evidence": "",
        }
        for cat in SIGNAL_CATEGORIES
    }


def default_extraction() -> dict[str, Any]:
    return {
        "conversation_id": None,
        "guest_text": "",
        "signals": default_signal_block(),
        "parser_confidence": 0.0,
        "notes": "",
    }


def validate_and_normalize_extraction(payload: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    """
    Validate extraction payload and normalize it to the canonical schema.

    Returns:
        (is_valid, normalized_payload, errors)
    """
    errors: list[str] = []
    normalized = default_extraction()

    if not isinstance(payload, dict):
        return False, normalized, ["payload is not a dict"]

    normalized["conversation_id"] = payload.get("conversation_id")
    guest_text = payload.get("guest_text", "")
    if not isinstance(guest_text, str):
        errors.append("guest_text must be a string")
        guest_text = str(guest_text)
    normalized["guest_text"] = guest_text.strip()

    parser_conf = payload.get("parser_confidence", 0.0)
    try:
        parser_conf = float(parser_conf)
    except (TypeError, ValueError):
        errors.append("parser_confidence must be numeric")
        parser_conf = 0.0
    normalized["parser_confidence"] = min(max(parser_conf, 0.0), 1.0)

    notes = payload.get("notes", "")
    normalized["notes"] = notes if isinstance(notes, str) else str(notes)

    raw_signals = payload.get("signals", {})
    if not isinstance(raw_signals, dict):
        errors.append("signals must be an object")
        raw_signals = {}

    for cat in SIGNAL_CATEGORIES:
        raw_block = raw_signals.get(cat, {})
        if not isinstance(raw_block, dict):
            errors.append(f"signals.{cat} must be an object")
            raw_block = {}

        detected = bool(raw_block.get("detected", False))
        conf_raw = raw_block.get("confidence", 0.0)
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            errors.append(f"signals.{cat}.confidence must be numeric")
            conf = 0.0
        conf = min(max(conf, 0.0), 1.0)

        evidence = raw_block.get("evidence", "")
        if not isinstance(evidence, str):
            evidence = str(evidence)
            errors.append(f"signals.{cat}.evidence must be a string")

        # If confidence is high but detected false, keep confidence but do not force detect.
        normalized["signals"][cat] = {
            "detected": detected,
            "confidence": conf,
            "evidence": evidence.strip(),
        }

    # Basic quality checks
    if not normalized["guest_text"]:
        errors.append("guest_text is empty")
    for cat in SIGNAL_CATEGORIES:
        if normalized["signals"][cat]["detected"] and not normalized["signals"][cat]["evidence"]:
            errors.append(f"signals.{cat}.evidence missing for detected signal")

    return len(errors) == 0, normalized, errors


def extraction_to_labels(extraction: dict[str, Any]) -> dict[str, int]:
    labels = {}
    for cat in SIGNAL_CATEGORIES:
        detected = extraction.get("signals", {}).get(cat, {}).get("detected", False)
        labels[cat] = int(bool(detected))
    return labels


def deep_copy_extraction(extraction: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(extraction)
