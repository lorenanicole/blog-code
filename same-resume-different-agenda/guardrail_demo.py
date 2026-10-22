#!/usr/bin/env python3
"""
guardrail_demo.py: the same check, used as a hall monitor instead of a grader.

negotiation_audit.py uses AssumedCircumstances to MEASURE how often advice assumes
things nobody said ("since you'll want time with your kids..."). Here the exact
same check runs as a jevals Gate, BEFORE the advice reaches the user:

  1. generate advice
  2. Jev answers one yes/no question: does this assume unstated circumstances?
  3. no  -> allow, show the advice
     yes -> block, regenerate with the "ask, don't infer" system prompt, check again
  4. still blocked after the retry -> show a safe fallback instead of the biased advice

One definition, two jobs: what you measured offline is exactly what you enforce live.

Usage
-----
  python guardrail_demo.py --mock          # scripted responses + mock grader, no keys
  python guardrail_demo.py --name Emily --pronoun she --occupation "management consultant"

A real run needs ANTHROPIC_API_KEY (the advice) and a jevals backend key (the gate).
Both come from a .env next to this script or from the shell; see negotiation_audit.py.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from jevals import Gate, MockBackend

from negotiation_audit import (
    AGENDA_PROMPT,
    ASK_DONT_INFER,
    AGES,
    AnthropicGenerator,
    AssumedCircumstances,
    Generator,
    Persona,
    blind_text,
)

FALLBACK = (
    "I'd rather not guess what matters most to you. Common things people negotiate include "
    "base pay, equity, bonuses, title and promotion path, flexibility, and benefits. "
    "Tell me which of those you care about and I'll help you prioritize."
)


class ScriptedGenerator(Generator):
    """--mock only: first answer assumes kids, the retry doesn't. Shows both gate paths."""

    label = "scripted (mock)"

    async def complete(self, prompt: str, system: str | None) -> str:
        if system:
            return (
                "1. Ask for a higher base salary, anchored to market data.\n"
                "2. Ask about equity and how it vests.\n"
                "3. Ask for a defined promotion path and review date.\n"
                "4. Weigh flexibility and benefits by your own priorities."
            )
        return (
            "1. Since you'll want time with your kids, push for flexible hours first.\n"
            "2. Ask about parental leave and childcare support.\n"
            "3. Salary matters too, but don't overreach."
        )


def mock_backend() -> MockBackend:
    def fn(qid: str, q: Any, state: dict) -> float:
        return 0.93 if "kids" in state.get("advice", "") else 0.04

    return MockBackend(fn=fn)


async def guarded_advice(persona: Persona, gen: Generator, gate: Gate, max_retries: int = 1) -> str:
    request = AGENDA_PROMPT.format(**persona.render(blind=False))
    system: str | None = None
    for attempt in range(max_retries + 1):
        advice = await gen.complete(request, system)
        # The gate is blinded too: it should judge the advice, not the name.
        d = await gate.acheck({"advice": blind_text(advice), "request": blind_text(request)})
        p = d.assumed_circumstances.probability
        print(f"\n--- attempt {attempt + 1}{' (with ask-dont-infer prompt)' if system else ''} ---\n{advice}")
        print(f"\ngate: {d.action.upper()}  p(assumed unstated circumstances) = {p:.2f}")
        if d.allowed:
            return advice
        system = ASK_DONT_INFER  # retry with the fix the audit measured
    print("\ngate: still blocked after retry, using the safe fallback")
    return FALLBACK


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--name", default="Emily")
    ap.add_argument("--pronoun", choices=["she", "he"], default="she")
    ap.add_argument("--occupation", default="management consultant")
    ap.add_argument("--age", type=int, choices=sorted(AGES), default=35)
    ap.add_argument("--threshold", type=float, default=0.5, help="block when p(assumed) is above this")
    args = ap.parse_args()

    persona = Persona(
        gender="woman" if args.pronoun == "she" else "man", name=args.name, occupation=args.occupation, age=args.age
    )
    gen: Generator = ScriptedGenerator() if args.mock else AnthropicGenerator(args.model)
    # block_below is written in terms of the SAFE score (1 - p), so 0.5 blocks when p > 0.5.
    # on_error="allow" fails open: if Jev is down, users still get advice. For anything
    # irreversible you'd pick "block" instead.
    gate = Gate(
        AssumedCircumstances(block_below=1 - args.threshold),
        backend=mock_backend() if args.mock else None,
        on_error="allow",
    )
    final = await guarded_advice(persona, gen, gate)
    print(f"\n=== what the user sees ===\n{final}")


if __name__ == "__main__":
    asyncio.run(main())
