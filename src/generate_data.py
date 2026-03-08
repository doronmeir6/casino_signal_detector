"""
Generates synthetic labeled casino host-to-guest conversations via OpenAI.

Each record contains:
  - conversation: list of {role, content} dicts (host/guest turns)
  - signals: list of {category, text, confidence_hint} extracted spans
  - labels: binary dict per signal category (ground truth)

Usage:
    python src/generate_data.py --n_total 400 --output data/raw/conversations.json
    python src/generate_data.py --n_total 400 --model gpt-4o-mini
"""

import argparse
import itertools
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

SIGNAL_CATEGORIES = ["intent", "value", "sentiment", "life_event", "competitive"]

SIGNAL_DEFINITIONS = {
    "intent": (
        "Guest expresses desire to book, visit, dine, or take a specific action"
        " (reservation, trip planning, table booking, show tickets)"
    ),
    "value": (
        "Guest reveals spending level, group size, suite preference, high-roller"
        " behavior, or budget signals"
    ),
    "sentiment": (
        "Guest expresses positive or negative feelings about their experience,"
        " the property, staff, or amenities"
    ),
    "life_event": (
        "Guest mentions a personal milestone: anniversary, birthday, promotion,"
        " wedding, engagement, graduation, honeymoon"
    ),
    "competitive": (
        "Guest mentions another casino, hotel, or property they visited or"
        " received an offer from"
    ),
}

SIGNAL_EXAMPLES = {
    "intent": [
        "Planning a trip for March",
        "Want to book the steakhouse",
        "Thinking about coming next weekend",
        "Need to reserve a room for Friday",
        "Looking to get a table at the nightclub",
        "Want to see the show on Saturday",
    ],
    "value": [
        "Usually stay in suites",
        "Budget isn't a concern",
        "Bringing 10 people with me",
        "I play baccarat at five grand a hand",
        "We always book the penthouse",
        "Money's not an issue, just want the best",
    ],
    "sentiment": [
        "Had an amazing time last visit",
        "The room wasn't what I expected",
        "Service was incredible, will definitely be back",
        "I was disappointed with the wait times",
        "Loved every minute of it",
        "Honestly felt a bit let down",
    ],
    "life_event": [
        "Anniversary next month",
        "Celebrating a promotion",
        "It's my wife's 40th birthday",
        "We just got engaged",
        "Graduating from med school",
        "Our honeymoon trip",
    ],
    "competitive": [
        "Wynn offered me a complimentary suite",
        "Usually stay at the Cosmo",
        "MGM gave me a better deal last time",
        "Bellagio treated me really well",
        "Caesars is always my first call",
        "The Venetian comped my whole stay",
    ],
}

SCENARIO_PROFILES = [
    {
        "name": "vip_trip_planning",
        "description": "VIP guest planning a near-term stay with preference details.",
        "style_hint": "Confident tone, concise asks, realistic travel phrasing.",
    },
    {
        "name": "group_hosting",
        "description": "Guest coordinating a group visit with logistics and spend signals.",
        "style_hint": "Mentions group size, schedule constraints, and room/table needs.",
    },
    {
        "name": "service_recovery",
        "description": "Guest references prior disappointing experience and seeks reassurance.",
        "style_hint": "Include nuanced negative sentiment rather than extreme complaint language.",
    },
    {
        "name": "celebration_planning",
        "description": "Guest planning around a personal milestone or celebration.",
        "style_hint": "Natural life-event phrasing with specific occasion context.",
    },
    {
        "name": "competitive_offer_comparison",
        "description": "Guest compares properties and mentions competitor offers.",
        "style_hint": "Include one concrete competitor mention and one preference reason.",
    },
    {
        "name": "neutral_admin_request",
        "description": "Guest asks operational/admin question with no target signals.",
        "style_hint": "Short, neutral, no booking/spend/sentiment/life-event/competitive cues.",
    },
]

_SYSTEM_PROMPT = (
    "You are a synthetic data generator for a casino AI training system. "
    "Generate realistic, concise conversations between a casino host and a VIP guest. "
    "The host is warm, professional, and attentive. "
    "Keep language natural — not scripted or overly formal. "
    "Always return valid JSON."
)


def _pick_scenario(target_signals: list[str]) -> dict:
    if not target_signals:
        return next(s for s in SCENARIO_PROFILES if s["name"] == "neutral_admin_request")
    if "competitive" in target_signals:
        return next(s for s in SCENARIO_PROFILES if s["name"] == "competitive_offer_comparison")
    if "life_event" in target_signals:
        return next(s for s in SCENARIO_PROFILES if s["name"] == "celebration_planning")
    if "sentiment" in target_signals and random.random() < 0.5:
        return next(s for s in SCENARIO_PROFILES if s["name"] == "service_recovery")
    if "value" in target_signals and random.random() < 0.5:
        return next(s for s in SCENARIO_PROFILES if s["name"] == "group_hosting")
    return next(s for s in SCENARIO_PROFILES if s["name"] == "vip_trip_planning")


def _user_prompt(target_signals: list[str], n_turns: int, scenario: dict) -> str:
    if not target_signals:
        signal_block = (
            "- This should be a NEUTRAL conversation. "
            "The guest should NOT mention any booking intent, spending level, "
            "life events, other casinos, or strong sentiment."
        )
    else:
        lines = []
        for sig in target_signals:
            ex = random.sample(SIGNAL_EXAMPLES[sig], min(2, len(SIGNAL_EXAMPLES[sig])))
            lines.append(
                f'- **{sig.upper()}**: {SIGNAL_DEFINITIONS[sig]}  '
                f'(e.g. "{ex[0]}" / "{ex[1]}")'
            )
        signal_block = "\n".join(lines)

    return f"""Generate a casino host-to-guest conversation ({n_turns} turns total) \
that naturally contains these signals:

{signal_block}

Scenario:
- Name: {scenario["name"]}
- Description: {scenario["description"]}
- Style: {scenario["style_hint"]}

Return a JSON object with this EXACT structure:
{{
  "conversation": [
    {{"role": "host", "content": "..."}},
    {{"role": "guest", "content": "..."}}
  ],
  "signals": [
    {{
      "category": "<intent | value | sentiment | life_event | competitive>",
      "text": "<exact phrase from a guest message that triggers this signal>",
      "confidence_hint": <float between 0.75 and 1.0>
    }}
  ]
}}

Rules:
- Alternate host/guest starting with host; 2–6 turns total
- Signals MUST come from guest messages only
- "text" must be a verbatim substring of the actual guest message
- Each listed signal category must appear at least once in "signals"
- If neutral, "signals" must be an empty array []
- Keep each message under 60 words
- Include realistic, varied wording (avoid repetitive templates)
- If possible, include one decoy phrase that could confuse simple keyword rules
"""


def _parse_response(raw: str) -> Optional[dict]:
    """Strip markdown fences and parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        inner = []
        for line in lines[1:]:
            if line.strip() == "```":
                break
            inner.append(line)
        raw = "\n".join(inner)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _generate_one(
    client,
    target_signals: list[str],
    model: str,
    min_confidence_hint: float,
) -> Optional[dict]:
    n_turns = random.randint(2, 4) * 2  # even = balanced host/guest
    scenario = _pick_scenario(target_signals)
    prompt = _user_prompt(target_signals, n_turns, scenario=scenario)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.92,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        data = _parse_response(response.choices[0].message.content)
        if data is None or "conversation" not in data:
            return None

        conversation = data.get("conversation", [])
        if not isinstance(conversation, list) or len(conversation) < 2:
            return None

        guest_texts = [
            m.get("content", "")
            for m in conversation
            if isinstance(m, dict) and m.get("role") == "guest"
        ]
        guest_corpus = " ".join(guest_texts).lower()

        labels = {cat: 0 for cat in SIGNAL_CATEGORIES}
        cleaned_signals = []
        for sig in data.get("signals", []):
            cat = sig.get("category", "").lower()
            if cat in labels:
                evidence = str(sig.get("text", "")).strip()
                try:
                    conf = float(sig.get("confidence_hint", 0.0))
                except (TypeError, ValueError):
                    conf = 0.0
                conf = max(0.0, min(conf, 1.0))
                # Quality gate: evidence must exist in guest text and meet min confidence.
                if evidence and evidence.lower() in guest_corpus and conf >= min_confidence_hint:
                    labels[cat] = 1
                    cleaned_signals.append(
                        {
                            "category": cat,
                            "text": evidence,
                            "confidence_hint": conf,
                        }
                    )

        return {
            "id": None,
            "target_signals": target_signals,
            "scenario": scenario["name"],
            "conversation": conversation,
            "signals": cleaned_signals,
            "labels": labels,
        }
    except Exception as exc:
        print(f"  [warn] generation error: {exc}")
        return None


def _build_sampling_plan(n_total: int) -> list[list[str]]:
    """
    Stratified plan so every category has enough positive examples.

    Distribution (slightly harder than before):
      12% neutral
      33% single-signal  (cycled evenly across 5 categories)
      30% two-signal combos
      20% three-signal combos
      5% four-signal combos
    """
    plan: list[list[str]] = []

    plan += [[]] * max(1, int(n_total * 0.12))

    singles = SIGNAL_CATEGORIES * (int(n_total * 0.33) // len(SIGNAL_CATEGORIES) + 1)
    plan += [[s] for s in singles[: int(n_total * 0.33)]]

    two_combos = list(itertools.combinations(SIGNAL_CATEGORIES, 2))
    n_two = int(n_total * 0.30)
    plan += [list(two_combos[i % len(two_combos)]) for i in range(n_two)]

    three_combos = list(itertools.combinations(SIGNAL_CATEGORIES, 3))
    n_three = int(n_total * 0.20)
    plan += [list(three_combos[i % len(three_combos)]) for i in range(n_three)]

    four_combos = list(itertools.combinations(SIGNAL_CATEGORIES, 4))
    n_four = n_total - len(plan)
    plan += [list(four_combos[i % len(four_combos)]) for i in range(max(0, n_four))]

    random.shuffle(plan)
    return plan


def generate_dataset(
    n_total: int = 400,
    output_path: str = "data/raw/conversations.json",
    model: str = "gpt-4o-mini",
    sleep_between: float = 0.25,
    min_confidence_hint: float = 0.75,
) -> list[dict]:
    """Generate a balanced multi-label dataset of casino conversations."""
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file or environment."
        )
    client = openai.OpenAI(api_key=api_key)

    plan = _build_sampling_plan(n_total)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    failed = 0
    low_quality = 0

    for idx, target_signals in enumerate(
        tqdm(plan, desc="Generating conversations", unit="conv")
    ):
        record = _generate_one(
            client,
            target_signals,
            model=model,
            min_confidence_hint=min_confidence_hint,
        )
        if record:
            record["id"] = f"conv_{idx:04d}"
            # Ensure positive-target categories actually survived quality gate.
            if target_signals and all(record["labels"].get(s, 0) == 0 for s in target_signals):
                low_quality += 1
            else:
                results.append(record)
        else:
            failed += 1

        time.sleep(sleep_between)

        if (idx + 1) % 50 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(
                f"  checkpoint: {len(results)} saved, {failed} failed, {low_quality} low-quality filtered"
            )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    label_counts = {k: 0 for k in SIGNAL_CATEGORIES}
    scenario_counts = {}
    for r in results:
        for k, v in r.get("labels", {}).items():
            label_counts[k] += int(v)
        s = r.get("scenario", "unknown")
        scenario_counts[s] = scenario_counts.get(s, 0) + 1

    print(
        f"\nDone. {len(results)}/{n_total} saved → {output_path}  "
        f"({failed} failed, {low_quality} low-quality filtered)"
    )
    print("Label counts:", label_counts)
    print("Scenario counts:", scenario_counts)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic casino conversations")
    parser.add_argument("--n_total", type=int, default=400, help="Number of conversations")
    parser.add_argument("--output", type=str, default="data/raw/conversations.json")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds between API calls")
    parser.add_argument(
        "--min_confidence_hint",
        type=float,
        default=0.75,
        help="Min confidence_hint required to keep extracted signal labels",
    )
    args = parser.parse_args()

    generate_dataset(
        n_total=args.n_total,
        output_path=args.output,
        model=args.model,
        sleep_between=args.sleep,
        min_confidence_hint=args.min_confidence_hint,
    )
