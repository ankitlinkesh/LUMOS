"""``build_prompt`` must match PoisonedRAG's own template byte-for-byte --
loads the actual paper source at test time, so drift in either file breaks
the build instead of silently diverging."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from triad.pipeline import MULTIPLE_PROMPT, build_prompt

PROMPTS_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "PoisonedRAG" / "src" / "prompts.py"


def _load_paper_prompts():
    spec = importlib.util.spec_from_file_location("_poisonedrag_prompts_under_test", PROMPTS_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_multiple_prompt_matches_the_paper_source_verbatim():
    paper = _load_paper_prompts()
    assert MULTIPLE_PROMPT == paper.MULTIPLE_PROMPT


def test_build_prompt_matches_wrap_prompt_id_4():
    paper = _load_paper_prompts()
    question = "who is the CEO?"
    contexts = ["passage one about the company.", "passage two about leadership.", "an adversarial passage."]

    ours = build_prompt(question, contexts)
    theirs = paper.wrap_prompt(question, contexts, prompt_id=4)

    assert ours == theirs


def test_build_prompt_single_context():
    paper = _load_paper_prompts()
    question = "what time is the meeting?"
    contexts = ["the meeting is at 3pm on Friday."]

    assert build_prompt(question, contexts) == paper.wrap_prompt(question, contexts, prompt_id=4)
