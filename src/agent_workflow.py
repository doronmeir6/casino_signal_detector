"""
Explicit 3-agent workflow for assignment presentation:
  1) ExtractorAgent  -> structured signal extraction from conversation
  2) ValidatorAgent  -> quality checks / fallback decisions
  3) ScorerAgent     -> XGBoost confidence scoring

This follows a deterministic workflow pattern (not autonomous group-chat),
which is easier to evaluate and debug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.infer_extract_llm import LLMConversationExtractor
from src.predict import SignalDetector
from src.signal_schema import SIGNAL_CATEGORIES, validate_and_normalize_extraction


@dataclass
class WorkflowConfig:
    model_dir: str = "models"
    llm_model: str = "gpt-4o-mini"
    min_parser_confidence: float = 0.45
    threshold: float = 0.45


class ExtractorAgent:
    def __init__(self, cfg: WorkflowConfig):
        self.extractor = LLMConversationExtractor(model=cfg.llm_model)

    def run(self, conversation: list[dict], conversation_id: Optional[str] = None) -> dict:
        return self.extractor.extract(conversation, conversation_id=conversation_id)


class ValidatorAgent:
    def __init__(self, cfg: WorkflowConfig):
        self.min_parser_confidence = cfg.min_parser_confidence

    def run(self, extraction: dict) -> tuple[bool, dict, list[str]]:
        ok, normalized, errors = validate_and_normalize_extraction(extraction)
        parser_conf = float(normalized.get("parser_confidence", 0.0) or 0.0)
        if parser_conf < self.min_parser_confidence:
            errors.append("parser_confidence below threshold")
            ok = False

        # Light extra guard: detected signals should have evidence.
        for cat in SIGNAL_CATEGORIES:
            block = normalized.get("signals", {}).get(cat, {})
            if block.get("detected") and not (block.get("evidence") or "").strip():
                ok = False
                errors.append(f"{cat} detected without evidence")
        return ok, normalized, errors


class ScorerAgent:
    def __init__(self, cfg: WorkflowConfig):
        # use_llm_extractor=False: extraction is provided by ExtractorAgent
        self.detector = SignalDetector(
            model_dir=cfg.model_dir,
            threshold=cfg.threshold,
            use_llm_extractor=False,
        )

    def run(self, extraction: dict) -> list[dict]:
        return self.detector.detect_from_extraction(extraction, include_evidence=True)


class SignalWorkflow:
    def __init__(self, cfg: WorkflowConfig):
        self.cfg = cfg
        self.extractor = ExtractorAgent(cfg)
        self.validator = ValidatorAgent(cfg)
        self.scorer = ScorerAgent(cfg)

    def run(self, conversation: list[dict], conversation_id: Optional[str] = None) -> dict[str, Any]:
        raw_extraction = self.extractor.run(conversation, conversation_id=conversation_id)
        is_valid, extraction, errors = self.validator.run(raw_extraction)
        detections = self.scorer.run(extraction) if is_valid else []
        return {
            "conversation_id": conversation_id,
            "is_valid_extraction": is_valid,
            "validation_errors": errors,
            "extraction": extraction,
            "detections": detections,
        }


if __name__ == "__main__":
    demo = [
        {"role": "host", "content": "Hi! How can I help?"},
        {"role": "guest", "content": "I might visit next month. Wynn offered me a suite, but I prefer your service."},
    ]
    workflow = SignalWorkflow(WorkflowConfig())
    result = workflow.run(demo, conversation_id="demo_001")
    print(result)
