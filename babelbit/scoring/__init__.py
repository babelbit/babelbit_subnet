"""Scoring utilities shared by runner tests and CLI helpers."""

# Re-export score_jsonl for compatibility with existing imports.
from .score_dialogue import score_jsonl  # noqa: F401

__all__ = ["score_jsonl"]
