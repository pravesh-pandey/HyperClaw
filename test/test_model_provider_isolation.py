"""Tests that selecting a model provider returns only models for that provider.

Regression guard for the issue where:
- Selecting Codex should show ONLY GPT/Codex models (never Claude models)
- Selecting Claude Code should show ONLY Claude models (never Codex/GPT models)

The dispatch lives in api_models() → _slot_backend() → _cc_models() /
_codex_models(). These tests verify the isolation contract at each layer.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro_crew.acp_backends import (
    ACP_BACKEND_CLAUDE,
    ACP_BACKEND_CODEX,
    ACP_BACKEND_KIRO,
)
from kiro_crew.dashboard.handlers.agents import (
    _advertised_cc_models,
    _advertised_codex_models,
    _cc_models,
    _codex_models,
    _slot_backend,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _request_with_providers(providers: dict) -> MagicMock:
    """Fake aiohttp request whose sessions.active_providers() yields ``providers``."""
    sessions = SimpleNamespace(active_providers=lambda: list(providers.values()))
    state = SimpleNamespace(sessions=sessions)
    req = MagicMock()
    req.app.__getitem__.return_value = state
    return req


def _request_no_state() -> MagicMock:
    """Fake request with no state key (simulates a cold / uninitialized gateway)."""
    req = MagicMock()
    req.app.__getitem__.side_effect = KeyError("state")
    return req


class _FakeProvider:
    """Minimal provider double with a configurable backend tag."""

    def __init__(self, models: list[dict], backend: str = ACP_BACKEND_CLAUDE):
        self._models = models
        self.client = SimpleNamespace(backend=backend)

    def available_models(self) -> list[dict]:
        return self._models


# ─────────────────────────────────────────────────────────────────────────────
# GPT / Codex model names that must NEVER appear when Claude Code is selected
# ─────────────────────────────────────────────────────────────────────────────

_GPT_MODEL_IDS = [
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "codex-model-from-adapter",
    "o3",
    "o4-mini",
]

# Claude model names that must NEVER appear when Codex is selected
_CLAUDE_MODEL_IDS = [
    "opus-4.8-1m",
    "opus-4.8",
    "sonnet-4.6-1m",
    "haiku-4.5",
    "fable-5-1m",
    "claude-opus-4.8",
    "global.anthropic.claude-opus-4-8[1m]",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. _codex_models: must never contain Claude model ids
# ─────────────────────────────────────────────────────────────────────────────


class TestCodexModelsContainNoClaudeModels:
    """When the Codex backend is selected, the returned model list must not
    include any Anthropic/Claude model ids."""

    def test_cold_codex_list_has_no_claude_models(self):
        """A cold gateway (no live session) returns only ['auto'] — not any Claude id."""
        out = _codex_models(_request_with_providers({}))
        names = {m["model_name"] for m in out}
        for claude_id in _CLAUDE_MODEL_IDS:
            assert (
                claude_id not in names
            ), f"Claude model {claude_id!r} must not appear in Codex picker"

    def test_cold_codex_list_contains_only_auto(self):
        """With no live Codex session the picker contains exactly one row: auto."""
        out = _codex_models(_request_with_providers({}))
        assert [m["model_name"] for m in out] == ["auto"]

    def test_live_codex_session_adds_gpt_rows_not_claude_rows(self):
        """A live Codex session merges its GPT rows; Claude rows must not appear."""
        provider = _FakeProvider(
            [
                {"modelId": "gpt-5.6-sol", "name": "GPT 5.6 Sol", "description": ""},
                {"modelId": "o3", "name": "o3", "description": ""},
            ],
            ACP_BACKEND_CODEX,
        )
        out = _codex_models(_request_with_providers({"s": provider}))
        names = {m["model_name"] for m in out}

        assert "gpt-5.6-sol" in names
        assert "o3" in names

        for claude_id in _CLAUDE_MODEL_IDS:
            assert (
                claude_id not in names
            ), f"Claude model {claude_id!r} must not appear in Codex picker"

    def test_codex_advertised_ignores_non_codex_providers(self):
        """_advertised_codex_models must ignore Claude/Kiro providers in the
        session pool — even if they expose available_models()."""
        claude_provider = _FakeProvider(
            [{"modelId": "global.anthropic.claude-opus-4-8[1m]", "name": "Opus 4.8"}],
            ACP_BACKEND_CLAUDE,  # wrong backend
        )
        kiro_provider = _FakeProvider(
            [{"modelId": "claude-opus-4.8", "name": "Opus"}],
            ACP_BACKEND_KIRO,
        )
        providers = {"c": claude_provider, "k": kiro_provider}

        advertised = _advertised_codex_models(_request_with_providers(providers))
        names = {r["model_name"] for r in advertised}

        for claude_id in _CLAUDE_MODEL_IDS:
            assert claude_id not in names

    def test_codex_auto_sentinel_is_first(self):
        """The 'auto' sentinel always leads the Codex list, never hidden."""
        provider = _FakeProvider(
            [{"modelId": "gpt-5.6-sol", "name": "GPT 5.6 Sol"}],
            ACP_BACKEND_CODEX,
        )
        out = _codex_models(_request_with_providers({"s": provider}))
        assert out[0]["model_name"] == "auto"

    def test_codex_no_duplicate_auto(self):
        """'auto' must appear exactly once even if the adapter advertises it."""
        provider = _FakeProvider(
            [
                {"modelId": "auto", "name": "Auto"},
                {"modelId": "gpt-5.6-sol", "name": "GPT 5.6 Sol"},
            ],
            ACP_BACKEND_CODEX,
        )
        out = _codex_models(_request_with_providers({"s": provider}))
        assert [m["model_name"] for m in out].count("auto") == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. _cc_models: must never contain GPT/Codex model ids
# ─────────────────────────────────────────────────────────────────────────────


class TestCcModelsContainNoCodexModels:
    """When the Claude Code backend is selected, the returned model list must
    not include any Codex/GPT model ids."""

    def test_cold_cc_list_has_no_gpt_models(self):
        """A cold gateway returns only ['auto'] for Claude Code — not any GPT id."""
        out = _cc_models(_request_with_providers({}))
        names = {m["model_name"] for m in out}
        for gpt_id in _GPT_MODEL_IDS:
            assert (
                gpt_id not in names
            ), f"GPT model {gpt_id!r} must not appear in Claude Code picker"

    def test_cold_cc_list_contains_only_auto(self):
        out = _cc_models(_request_with_providers({}))
        assert [m["model_name"] for m in out] == ["auto"]

    def test_live_cc_session_adds_claude_rows_not_gpt_rows(self):
        """A live Claude Code session merges its Claude rows; GPT rows must not appear."""
        provider = _FakeProvider(
            [
                {"modelId": "global.anthropic.claude-opus-4-8[1m]", "name": "Opus 4.8"},
                {"modelId": "global.anthropic.claude-sonnet-4-6[1m]", "name": "Sonnet 4.6"},
            ],
            ACP_BACKEND_CLAUDE,
        )
        out = _cc_models(_request_with_providers({"s": provider}))
        names = {m["model_name"] for m in out}

        # Should contain Claude models, as the ids the adapter advertised --
        # those are the values its ``set_config_option`` accepts.
        assert "global.anthropic.claude-opus-4-8[1m]" in names
        assert "global.anthropic.claude-sonnet-4-6[1m]" in names

        for gpt_id in _GPT_MODEL_IDS:
            assert (
                gpt_id not in names
            ), f"GPT model {gpt_id!r} must not appear in Claude Code picker"

    def test_cc_advertised_ignores_codex_providers(self):
        """_advertised_cc_models must ignore Codex providers in the session pool."""
        codex_provider = _FakeProvider(
            [{"modelId": "gpt-5.6-sol", "name": "GPT 5.6 Sol"}],
            ACP_BACKEND_CODEX,  # wrong backend — must be ignored
        )
        advertised = _advertised_cc_models(_request_with_providers({"s": codex_provider}))
        names = {r["model_name"] for r in advertised}
        for gpt_id in _GPT_MODEL_IDS:
            assert gpt_id not in names

    def test_cc_auto_sentinel_is_first(self):
        """The 'auto' sentinel always leads the Claude Code list."""
        provider = _FakeProvider(
            [{"modelId": "global.anthropic.claude-opus-4-8[1m]", "name": "Opus 4.8"}],
            ACP_BACKEND_CLAUDE,
        )
        out = _cc_models(_request_with_providers({"s": provider}))
        assert out[0]["model_name"] == "auto"

    def test_cc_no_duplicate_auto(self):
        """'auto' must appear exactly once."""
        provider = _FakeProvider(
            [
                {"modelId": "global.anthropic.claude-opus-4-8[1m]", "name": "Opus 4.8"},
            ],
            ACP_BACKEND_CLAUDE,
        )
        out = _cc_models(_request_with_providers({"s": provider}))
        assert [m["model_name"] for m in out].count("auto") == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cross-contamination: mixed sessions in the pool
# ─────────────────────────────────────────────────────────────────────────────


class TestProviderPoolIsolation:
    """When multiple backend sessions are live simultaneously the pickers must
    read ONLY from sessions that match their own backend tag."""

    def test_codex_picker_ignores_claude_session_in_mixed_pool(self):
        """A pool with both Claude and Codex sessions: Codex picker reads only Codex."""
        claude_session = _FakeProvider(
            [
                {"modelId": "global.anthropic.claude-opus-4-8[1m]", "name": "Opus 4.8"},
                {"modelId": "global.anthropic.claude-sonnet-4-6[1m]", "name": "Sonnet 4.6"},
            ],
            ACP_BACKEND_CLAUDE,
        )
        codex_session = _FakeProvider(
            [
                {"modelId": "gpt-5.6-sol", "name": "GPT 5.6 Sol"},
                {"modelId": "o3", "name": "o3"},
            ],
            ACP_BACKEND_CODEX,
        )
        out = _codex_models(
            _request_with_providers({"claude": claude_session, "codex": codex_session})
        )
        names = {m["model_name"] for m in out}

        # Codex models are present
        assert "gpt-5.6-sol" in names
        assert "o3" in names

        # Claude models are absent
        for claude_id in _CLAUDE_MODEL_IDS:
            assert (
                claude_id not in names
            ), f"Claude {claude_id!r} leaked into Codex picker from mixed pool"

    def test_cc_picker_ignores_codex_session_in_mixed_pool(self):
        """A pool with both Claude and Codex sessions: CC picker reads only Claude."""
        claude_session = _FakeProvider(
            [
                {"modelId": "global.anthropic.claude-opus-4-8[1m]", "name": "Opus 4.8"},
            ],
            ACP_BACKEND_CLAUDE,
        )
        codex_session = _FakeProvider(
            [
                {"modelId": "gpt-5.6-sol", "name": "GPT 5.6"},
                {"modelId": "codex-model-from-adapter", "name": "Codex Model"},
            ],
            ACP_BACKEND_CODEX,
        )
        out = _cc_models(
            _request_with_providers({"claude": claude_session, "codex": codex_session})
        )
        names = {m["model_name"] for m in out}

        # Claude models are present, as advertised
        assert "global.anthropic.claude-opus-4-8[1m]" in names

        # Codex/GPT models are absent
        for gpt_id in _GPT_MODEL_IDS:
            assert (
                gpt_id not in names
            ), f"GPT {gpt_id!r} leaked into Claude Code picker from mixed pool"

    def test_codex_picker_ignores_kiro_session(self):
        """Kiro (ACP_BACKEND_KIRO / '') sessions must be ignored by the Codex picker."""
        kiro_session = _FakeProvider(
            [{"modelId": "claude-opus-4.8", "name": "Opus"}],
            ACP_BACKEND_KIRO,
        )
        codex_session = _FakeProvider(
            [{"modelId": "gpt-5.6-sol", "name": "GPT 5.6"}],
            ACP_BACKEND_CODEX,
        )
        out = _codex_models(_request_with_providers({"kiro": kiro_session, "codex": codex_session}))
        names = {m["model_name"] for m in out}
        assert "claude-opus-4.8" not in names
        assert "gpt-5.6-sol" in names


# ─────────────────────────────────────────────────────────────────────────────
# 4. api_models dispatch routes to the correct picker
# ─────────────────────────────────────────────────────────────────────────────


class TestApiModelsBackendDispatch:
    """The api_models() endpoint must call the right picker and return only the
    models for the selected backend — no cross-backend bleed."""

    @pytest.mark.asyncio
    async def test_codex_backend_returns_codex_models_only(self, tmp_path):
        """With backend=codex the endpoint must not return any Claude model rows."""
        from kiro_crew.dashboard.handlers.agents import api_models

        provider = _FakeProvider(
            [{"modelId": "gpt-5.6-sol", "name": "GPT 5.6 Sol"}],
            ACP_BACKEND_CODEX,
        )
        sessions = SimpleNamespace(active_providers=lambda: [provider])
        state = SimpleNamespace(sessions=sessions)

        request = MagicMock()
        request.app.__getitem__.return_value = state
        request.query = {"backend": ACP_BACKEND_CODEX}

        mock_cfg = MagicMock()
        mock_cfg.agent.acp_backend = ACP_BACKEND_CODEX
        mock_cfg.agent.model = "auto"
        mock_cfg.agent.sandbox = "auto"

        with patch(
            "kiro_crew.dashboard.handlers.agents.KiroCrewConfig.load",
            new_callable=AsyncMock if False else MagicMock,
            return_value=mock_cfg,
        ):
            with patch("asyncio.to_thread", return_value=mock_cfg):
                response = await api_models(request)

        data = response.body if hasattr(response, "body") else None
        if data is None:
            # json_response stores the content in _payload or text
            import json

            text = response.text if hasattr(response, "text") else ""
            rows = json.loads(text) if text else []
        else:
            import json

            rows = json.loads(data)

        names = {m["model_name"] for m in rows}
        for claude_id in _CLAUDE_MODEL_IDS:
            assert (
                claude_id not in names
            ), f"Claude {claude_id!r} leaked into Codex api_models response"

    @pytest.mark.asyncio
    async def test_claude_backend_returns_cc_models_only(self, tmp_path):
        """With backend=claude the endpoint must not return any GPT/Codex model rows."""
        from kiro_crew.dashboard.handlers.agents import api_models

        provider = _FakeProvider(
            [{"modelId": "global.anthropic.claude-opus-4-8[1m]", "name": "Opus 4.8"}],
            ACP_BACKEND_CLAUDE,
        )
        sessions = SimpleNamespace(active_providers=lambda: [provider])
        state = SimpleNamespace(sessions=sessions)

        request = MagicMock()
        request.app.__getitem__.return_value = state
        request.query = {"backend": ACP_BACKEND_CLAUDE}

        mock_cfg = MagicMock()
        mock_cfg.agent.acp_backend = ACP_BACKEND_CLAUDE
        mock_cfg.agent.model = "auto"
        mock_cfg.agent.sandbox = "auto"

        with patch("asyncio.to_thread", return_value=mock_cfg):
            response = await api_models(request)

        import json

        text = response.text if hasattr(response, "text") else response.body.decode()
        rows = json.loads(text)
        names = {m["model_name"] for m in rows}

        for gpt_id in _GPT_MODEL_IDS:
            assert (
                gpt_id not in names
            ), f"GPT {gpt_id!r} leaked into Claude Code api_models response"


# ─────────────────────────────────────────────────────────────────────────────
# 5. _slot_backend selects the correct backend
# ─────────────────────────────────────────────────────────────────────────────


class TestSlotBackend:
    """_slot_backend must return the correct backend tag so api_models dispatches
    to the right picker."""

    def test_explicit_codex_backend_param(self):
        """?backend=codex always selects the Codex picker."""
        req = MagicMock()
        req.query = {"backend": ACP_BACKEND_CODEX}
        result = _slot_backend(req, ACP_BACKEND_CLAUDE)  # configured=claude, but explicit wins
        assert result == ACP_BACKEND_CODEX

    def test_explicit_claude_backend_param(self):
        """?backend=claude always selects the Claude Code picker."""
        req = MagicMock()
        req.query = {"backend": ACP_BACKEND_CLAUDE}
        result = _slot_backend(req, ACP_BACKEND_CODEX)  # configured=codex, but explicit wins
        assert result == ACP_BACKEND_CLAUDE

    def test_no_param_falls_back_to_configured_backend(self):
        """Without ?backend= or ?slot= the configured backend is used."""
        req = MagicMock()
        req.query = {}
        assert _slot_backend(req, ACP_BACKEND_CODEX) == ACP_BACKEND_CODEX
        assert _slot_backend(req, ACP_BACKEND_CLAUDE) == ACP_BACKEND_CLAUDE

    def test_slot_with_codex_session_overrides_configured_claude(self):
        """?slot=<key> reads the session's own backend, not the global config."""
        slot_obj = SimpleNamespace(acp_backend=ACP_BACKEND_CODEX)
        slots = {"chat-1": slot_obj}
        state = SimpleNamespace(_slots=slots)

        req = MagicMock()
        req.query = {"slot": "chat-1"}
        req.app.__getitem__.return_value = state

        # configured is claude, but the slot runs codex
        result = _slot_backend(req, ACP_BACKEND_CLAUDE)
        assert result == ACP_BACKEND_CODEX

    def test_slot_with_claude_session_overrides_configured_codex(self):
        """Symmetric: a Claude session wins over a Codex configured default."""
        slot_obj = SimpleNamespace(acp_backend=ACP_BACKEND_CLAUDE)
        slots = {"chat-1": slot_obj}
        state = SimpleNamespace(_slots=slots)

        req = MagicMock()
        req.query = {"slot": "chat-1"}
        req.app.__getitem__.return_value = state

        result = _slot_backend(req, ACP_BACKEND_CODEX)
        assert result == ACP_BACKEND_CLAUDE

    def test_unknown_slot_falls_back_to_configured(self):
        """A ?slot= that does not exist falls back to the configured backend."""
        state = SimpleNamespace(_slots={})
        req = MagicMock()
        req.query = {"slot": "nonexistent-slot"}
        req.app.__getitem__.return_value = state

        assert _slot_backend(req, ACP_BACKEND_CODEX) == ACP_BACKEND_CODEX
        assert _slot_backend(req, ACP_BACKEND_CLAUDE) == ACP_BACKEND_CLAUDE

    def test_invalid_explicit_backend_falls_back_to_kiro(self):
        """An unrecognized ?backend= value falls back to the kiro default (empty string)."""
        req = MagicMock()
        req.query = {"backend": "totally-unknown-backend-xyz"}
        result = _slot_backend(req, ACP_BACKEND_CLAUDE)
        # resolve_selected_backend normalises unknowns to ACP_BACKEND_KIRO
        assert result == ACP_BACKEND_KIRO


# ─────────────────────────────────────────────────────────────────────────────
# 6. Completeness: all picker models belong to the right family
# ─────────────────────────────────────────────────────────────────────────────


class TestModelFamilyCompleteness:
    """Spot-check that the full picker lists for each backend are internally
    consistent and do not contain obvious stragglers from the other family."""

    def _is_claude_model(self, name: str) -> bool:
        if name == "auto":
            return False
        return (
            "claude" in name.lower()
            or "opus" in name.lower()
            or "sonnet" in name.lower()
            or "haiku" in name.lower()
            or "fable" in name.lower()
            or name.startswith("global.anthropic.")
        )

    def _is_gpt_or_codex_model(self, name: str) -> bool:
        if name == "auto":
            return False
        return (
            "gpt" in name.lower()
            or "codex" in name.lower()
            or name.startswith("o3")
            or name.startswith("o4")
        )

    def test_codex_picker_contains_no_claude_family_names(self):
        """All non-auto model names in the Codex picker must not look like Claude."""
        provider = _FakeProvider(
            [
                {"modelId": "gpt-5.6-sol", "name": "GPT 5.6 Sol"},
                {"modelId": "o3", "name": "o3"},
                {"modelId": "codex-mini", "name": "Codex Mini"},
            ],
            ACP_BACKEND_CODEX,
        )
        out = _codex_models(_request_with_providers({"s": provider}))
        for row in out:
            name = row["model_name"]
            if name == "auto":
                continue
            assert not self._is_claude_model(
                name
            ), f"Claude-family name {name!r} found in Codex picker"

    def test_cc_picker_contains_no_gpt_family_names(self):
        """All non-auto model names in the CC picker must not look like GPT/Codex."""
        provider = _FakeProvider(
            [
                {"modelId": "global.anthropic.claude-opus-4-8[1m]", "name": "Opus 4.8"},
                {"modelId": "global.anthropic.claude-sonnet-4-6[1m]", "name": "Sonnet 4.6"},
            ],
            ACP_BACKEND_CLAUDE,
        )
        out = _cc_models(_request_with_providers({"s": provider}))
        for row in out:
            name = row["model_name"]
            if name == "auto":
                continue
            assert not self._is_gpt_or_codex_model(
                name
            ), f"GPT/Codex-family name {name!r} found in Claude Code picker"
