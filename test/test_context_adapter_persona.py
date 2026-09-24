"""An adapted harness is told its own name in the shared persona prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

from kiro_crew import context as ctx


@pytest.fixture
def builder(monkeypatch, tmp_path: Path) -> ctx.ContextBuilder:
    prompt = tmp_path / "prompt.md"
    prompt.write_text("You are Kiro. Run kiro-cli; kiro reads this.", encoding="utf-8")
    monkeypatch.setattr(ctx, "_prompt_path", lambda mode=None: prompt)
    b = ctx.ContextBuilder.__new__(ctx.ContextBuilder)
    b._bot_name = "Kiro"
    return b


def _resolve(builder: ctx.ContextBuilder, provider_type: str, is_cc: bool = False) -> str:
    return builder._resolve_agent_prompt(
        None,
        project=None,
        mode="",
        session_key=None,
        is_cc=is_cc,
        private_owner=False,
        session_start=True,
        provider_type=provider_type,
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("codex", "You are Codex. Run codex; codex reads this."),
        ("opencode", "You are OpenCode. Run opencode; opencode reads this."),
        ("claude_code", "You are Claude. Run claude code; claude reads this."),
    ],
)
def test_adapted_harness_is_named(builder, label: str, expected: str) -> None:
    assert _resolve(builder, label) == expected


@pytest.mark.parametrize("label", ["acp", "kas"])
def test_kiro_family_prompt_is_unchanged(builder, label: str) -> None:
    assert "Kiro" in _resolve(builder, label)


def test_is_cc_alone_still_rebrands_to_claude(builder) -> None:
    # Callers that pass only is_cc keep their existing behaviour.
    assert _resolve(builder, "", is_cc=True).startswith("You are Claude.")
