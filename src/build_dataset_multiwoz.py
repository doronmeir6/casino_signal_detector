"""
Build casino-style training data from MultiWOZ 2.2.

This script converts MultiWOZ dialogues into the same schema used by the
project's synthetic generator:
  - conversation: [{"role": "host"|"guest", "content": "..."}]
  - signals: [{"category", "text", "confidence_hint"}]
  - labels: {"intent", "value", "sentiment", "life_event", "competitive"}

Primary usage:
    python src/build_dataset_multiwoz.py \
        --multiwoz_dir data/MultiWOZ_2.2 \
        --output data/raw/conversations_multiwoz.json \
        --services hotel restaurant
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm

SIGNAL_CATEGORIES = ["intent", "value", "sentiment", "life_event", "competitive"]

INTENT_KEYWORDS = (
    "book",
    "booking",
    "reserve",
    "reservation",
    "find",
    "looking for",
    "need",
    "want",
    "stay",
    "trip",
    "visit",
    "dine",
    "restaurant",
    "hotel",
)
VALUE_KEYWORDS = (
    "expensive",
    "luxury",
    "premium",
    "suite",
    "5 star",
    "4 star",
    "high-end",
    "group",
    "people",
)
POS_SENTIMENT = (
    "great",
    "good",
    "excellent",
    "amazing",
    "wonderful",
    "perfect",
    "happy",
    "love",
)
NEG_SENTIMENT = (
    "bad",
    "terrible",
    "awful",
    "disappointed",
    "unhappy",
    "frustrated",
    "not good",
    "not happy",
    "poor",
)
LIFE_EVENT_KEYWORDS = (
    "anniversary",
    "birthday",
    "honeymoon",
    "promotion",
    "engaged",
    "engagement",
    "wedding",
    "graduation",
)
COMPETITIVE_KEYWORDS = (
    "wynn",
    "bellagio",
    "caesars",
    "cosmo",
    "cosmopolitan",
    "mgm",
    "venetian",
    "palazzo",
)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    text = text.lower()
    return any(k in text for k in keywords)


def _extract_relevant_frames(turn: dict, allowed_services: set[str]) -> list[dict]:
    frames = turn.get("frames", [])
    return [f for f in frames if f.get("service") in allowed_services]


def _intent_from_turn(turn: dict, allowed_services: set[str]) -> bool:
    utt = turn.get("utterance", "").lower()
    if _contains_any(utt, INTENT_KEYWORDS):
        return True

    for frame in _extract_relevant_frames(turn, allowed_services):
        state = frame.get("state", {})
        active_intent = state.get("active_intent", "NONE")
        if active_intent and active_intent != "NONE":
            return True
    return False


def _value_from_turn(turn: dict, allowed_services: set[str]) -> bool:
    utt = turn.get("utterance", "").lower()
    if _contains_any(utt, VALUE_KEYWORDS):
        return True

    # Lightweight proxies from slot values:
    # - pricerange expensive
    # - party size >= 3
    # - nights stay >= 3
    # - stars >= 4
    for frame in _extract_relevant_frames(turn, allowed_services):
        slot_values = frame.get("state", {}).get("slot_values", {})
        for slot, values in slot_values.items():
            for val in values:
                v = str(val).lower()
                if "expensive" in v:
                    return True
                if "star" in slot:
                    m = re.search(r"\d+", v)
                    if m and int(m.group()) >= 4:
                        return True
                if any(k in slot for k in ("book people", "people")):
                    m = re.search(r"\d+", v)
                    if m and int(m.group()) >= 3:
                        return True
                if any(k in slot for k in ("book stay", "stay")):
                    m = re.search(r"\d+", v)
                    if m and int(m.group()) >= 3:
                        return True
    return False


def _sentiment_from_turn(turn: dict) -> bool:
    utt = turn.get("utterance", "").lower()
    return _contains_any(utt, POS_SENTIMENT) or _contains_any(utt, NEG_SENTIMENT)


def _life_event_from_turn(turn: dict) -> bool:
    utt = turn.get("utterance", "").lower()
    return _contains_any(utt, LIFE_EVENT_KEYWORDS)


def _competitive_from_turn(turn: dict) -> bool:
    utt = turn.get("utterance", "").lower()
    return _contains_any(utt, COMPETITIVE_KEYWORDS)


def _build_signals(guest_turns: list[dict], labels: dict[str, int]) -> list[dict]:
    signals = []
    category_checks = {
        "intent": lambda t: _intent_from_turn(t, {"hotel", "restaurant"}),
        "value": lambda t: _value_from_turn(t, {"hotel", "restaurant"}),
        "sentiment": _sentiment_from_turn,
        "life_event": _life_event_from_turn,
        "competitive": _competitive_from_turn,
    }
    for cat in SIGNAL_CATEGORIES:
        if not labels.get(cat):
            continue
        extractor = category_checks[cat]
        evidence = next((t.get("utterance", "") for t in guest_turns if extractor(t)), "")
        if evidence:
            signals.append(
                {
                    "category": cat,
                    "text": evidence[:180],
                    "confidence_hint": 0.85,
                }
            )
    return signals


def _convert_dialogue(dialogue: dict, allowed_services: set[str], rec_id: int) -> Optional[dict]:
    services = set(dialogue.get("services", []))
    if services.isdisjoint(allowed_services):
        return None

    turns = dialogue.get("turns", [])
    conversation = []
    guest_turns = []

    labels = {k: 0 for k in SIGNAL_CATEGORIES}

    for turn in turns:
        speaker = turn.get("speaker")
        utterance = turn.get("utterance", "").strip()
        if not utterance:
            continue

        if speaker == "USER":
            role = "guest"
            guest_turns.append(turn)
            if _intent_from_turn(turn, allowed_services):
                labels["intent"] = 1
            if _value_from_turn(turn, allowed_services):
                labels["value"] = 1
            if _sentiment_from_turn(turn):
                labels["sentiment"] = 1
            if _life_event_from_turn(turn):
                labels["life_event"] = 1
            if _competitive_from_turn(turn):
                labels["competitive"] = 1
        elif speaker == "SYSTEM":
            role = "host"
        else:
            continue

        conversation.append({"role": role, "content": utterance})

    if not conversation:
        return None

    signals = _build_signals(guest_turns, labels)
    return {
        "id": f"mwz_{rec_id:06d}",
        "source": "multiwoz_2.2",
        "dialogue_id": dialogue.get("dialogue_id"),
        "services": sorted(list(services)),
        "conversation": conversation,
        "signals": signals,
        "labels": labels,
    }


def _iter_dialogues(multiwoz_dir: Path, splits: list[str]) -> list[dict]:
    files = []
    for split in splits:
        split_dir = multiwoz_dir / split
        files.extend(sorted(split_dir.glob("dialogues_*.json")))

    dialogues = []
    for fp in tqdm(files, desc="Reading MultiWOZ files", unit="file"):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
            dialogues.extend(data)
    return dialogues


def build_dataset(
    multiwoz_dir: str,
    output: str,
    services: list[str],
    splits: list[str],
    max_records: Optional[int] = None,
) -> list[dict]:
    multiwoz_dir = Path(multiwoz_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    allowed_services = set(services)
    dialogues = _iter_dialogues(multiwoz_dir, splits)

    records = []
    for i, dialogue in enumerate(tqdm(dialogues, desc="Converting dialogues", unit="dlg")):
        rec = _convert_dialogue(dialogue, allowed_services, rec_id=len(records))
        if rec is None:
            continue
        records.append(rec)
        if max_records and len(records) >= max_records:
            break

    with open(output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # Print concise stats
    label_counts = {k: 0 for k in SIGNAL_CATEGORIES}
    for r in records:
        for k, v in r["labels"].items():
            label_counts[k] += int(v)

    print(f"\nSaved {len(records)} records to {output}")
    print("Label positives:")
    for k in SIGNAL_CATEGORIES:
        print(f"  {k:12s}: {label_counts[k]}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build project dataset from MultiWOZ 2.2")
    parser.add_argument("--multiwoz_dir", default="data/MultiWOZ_2.2")
    parser.add_argument("--output", default="data/raw/conversations_multiwoz.json")
    parser.add_argument("--services", nargs="+", default=["hotel", "restaurant"])
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--max_records", type=int, default=None)
    args = parser.parse_args()

    build_dataset(
        multiwoz_dir=args.multiwoz_dir,
        output=args.output,
        services=args.services,
        splits=args.splits,
        max_records=args.max_records,
    )
