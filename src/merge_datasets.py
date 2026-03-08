"""
Merge multiple dataset JSON files into one training file.

Each input file must follow the project schema:
  {
    "id": "...",
    "conversation": [...],
    "signals": [...],
    "labels": {...}
  }
"""

import argparse
import json
from pathlib import Path

SIGNAL_CATEGORIES = ["intent", "value", "sentiment", "life_event", "competitive"]


def _load_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def merge_datasets(inputs: list[str], output: str, dedupe_by_text: bool = True) -> list[dict]:
    records = []
    seen = set()

    for inp in inputs:
        path = Path(inp)
        part = _load_records(path)
        for rec in part:
            conv = rec.get("conversation", [])
            guest_text = " ".join(
                m.get("content", "") for m in conv if m.get("role") == "guest"
            ).strip().lower()
            key = guest_text if dedupe_by_text else rec.get("id")
            if key in seen:
                continue
            seen.add(key)

            # Ensure labels has all categories
            labels = rec.get("labels", {})
            rec["labels"] = {k: int(labels.get(k, 0)) for k in SIGNAL_CATEGORIES}
            records.append(rec)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} merged records -> {out}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge conversation dataset JSON files")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input dataset JSON files (space-separated)",
    )
    parser.add_argument("--output", default="data/raw/conversations_merged.json")
    parser.add_argument("--no_dedupe", action="store_true")
    args = parser.parse_args()

    merge_datasets(
        inputs=args.inputs,
        output=args.output,
        dedupe_by_text=not args.no_dedupe,
    )
