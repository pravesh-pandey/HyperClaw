"""Per-session ACP harness selection: the slot field, its endpoint, and its gates.

The provider is chosen per chat session in the composer. Unlike model and effort
there is no live switch, so every accepted request resets the slot's session; the
tests below pin that, plus the two refusals (this build cannot serve it / this
machine has not installed it) and the per-backend model translation the switch
depends on.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.acp_backends import (
    ACP_BACKEND_CLAUDE,
    ACP_BACKEND_CODEX,
    ACP_BACKEND_KIRO,
    ACP_BACKENDS_OWN_MODEL_CATALOG,
)
from kiro_crew.agent_sdk import INSTALLED, MISSING, BackendInstallState
from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.dashboard.chat import api_chat_slot_backend
from kiro_crew.dashboard.chat_persistence import _validate_acp_backend
from kiro_crew.dashboard.handlers.agents import _slot_backend
from kiro_crew.dashboard.slot_projection import SlotProjection
from kiro_crew.dashboard.state import DashboardState, _ChatSlot
from kiro_crew.session import POOL_DECISIONS
from kiro_crew.session_allocation import _pool_blocking_backend

_ROUTE = "/api/chat/slots/{slot}/backend"


def _make_app(state: DashboardState) -> web.Application:
    app = web.Application()
    app["state"] = state
    app.router.add_post(_ROUTE, api_chat_slot_backend)
    return app


def _mock_state(slot: _ChatSlot | None = None) -> DashboardState:
    state = MagicMock(spec=DashboardState)
    state._slots = {}
    if slot:
        state._slots[slot.key] = slot
    state.push_slots_update = MagicMock()
    state.sessions = MagicMock()
    state.sessions.reset = AsyncMock()
    state.sessions.get_provider = MagicMock(return_value=None)
    return state


def _installed(backend: str) -> BackendInstallState:
    return BackendInstallState(backend, backend or "kiro", INSTALLED)


def _missing(backend: str) -> BackendInstallState:
    return BackendInstallState(backend, backend or "kiro", MISSING, ("codex-acp",), "npm i -g pkg")


class TestSlotBackendField:
    def test_new_slot_inherits_rather_than_pinning_kiro(self) -> None:
        # None, not "": a session that never touched the picker must FOLLOW the
        # configured default. Pinning "" here would silently strand every new
        # session on kiro-cli for an operator whose default is Codex.
        assert _ChatSlot("s").acp_backend is None

    def test_summary_exposes_the_harness(self) -> None:
        """The composer renders the active provider from the slot row."""
        slot = _ChatSlot("s")
        slot.acp_backend = ACP_BACKEND_CODEX
        summary = SlotProjection.to_dict(
            slot,
            include_check_status=False,
            source_links=[],
            prompt_roles=frozenset(),
            redact=lambda v: v,
            parse_options=lambda _v: [],
            strip_options=lambda v: v,
            parse_cls_meta=lambda _v: None,
            is_turn_interrupted=lambda _v: False,
            is_system_notice=lambda _a, _b: False,
            latest_transcript_ts=lambda *_a, **_k: None,
            strip_markdown_preview=lambda v: v,
            resolve_effective_agent=lambda a, _p: a,
            budget_source_links=lambda v: v,
            project_source_links=lambda v, _b: v,
        )
        assert summary["acp_backend"] == ACP_BACKEND_CODEX
        # Sits beside the two knobs it cascades into, so one read drives all
        # three composer controls.
        assert {"model", "reasoning_effort", "acp_backend"} <= set(summary)


class TestPersistedBackendIsRegated:
    """A persisted slot outlives the build that wrote it (harness-parity H3)."""

    def test_selectable_value_survives(self) -> None:
        assert _validate_acp_backend(ACP_BACKEND_CODEX) == ACP_BACKEND_CODEX

    def test_explicit_kiro_round_trips(self) -> None:
        # "" is a real persisted pick, not absence — kiro is the floor and needs
        # no registry check.
        assert _validate_acp_backend("") == ACP_BACKEND_KIRO

    def test_unknown_harness_degrades_to_kiro(self) -> None:
        assert _validate_acp_backend("harness-from-a-later-build") == ACP_BACKEND_KIRO

    @pytest.mark.parametrize("raw", [None, 7, [], {}])
    def test_non_string_degrades_without_raising(self, raw: object) -> None:
        assert _validate_acp_backend(raw) == ACP_BACKEND_KIRO


class TestWarmPoolBypass:
    """The pool is spawned from the CONFIGURED harness and serves only that one."""

    @staticmethod
    def _deps(configured: str) -> MagicMock:
        deps = MagicMock()
        cfg = MagicMock()
        cfg.agent.acp_backend = configured
        deps.load_config = MagicMock(return_value=cfg)
        return deps

    def test_no_preference_keeps_the_warm_start(self) -> None:
        # None reaches the factory as "inherit", which IS what the pool was
        # spawned with — so it stays claimable whatever the configured harness is.
        assert _pool_blocking_backend(self._deps(ACP_BACKEND_KIRO), None) is False
        assert _pool_blocking_backend(self._deps(ACP_BACKEND_CODEX), None) is False

    def test_same_harness_keeps_the_warm_start(self) -> None:
        deps = self._deps(ACP_BACKEND_KIRO)
        assert _pool_blocking_backend(deps, ACP_BACKEND_KIRO) is False

    def test_different_harness_bypasses(self) -> None:
        deps = self._deps(ACP_BACKEND_KIRO)
        assert _pool_blocking_backend(deps, ACP_BACKEND_CODEX) is True

    def test_unreadable_config_bypasses_rather_than_claiming(self) -> None:
        deps = MagicMock()
        deps.load_config = MagicMock(side_effect=OSError("config unreadable"))
        assert _pool_blocking_backend(deps, ACP_BACKEND_CODEX) is True

    def test_decision_label_is_in_the_closed_metric_set(self) -> None:
        # The counter's cardinality is bounded by a frozen set; a label added to
        # the allocator and not to that set is dropped as "other".
        assert "bypass_backend" in POOL_DECISIONS


class TestBackendEndpoint:
    @pytest.mark.asyncio
    async def test_switch_resets_the_session_and_clears_the_model(self) -> None:
        slot = _ChatSlot("test")
        slot.model = "claude-opus-4.5"
        state = _mock_state(slot)
        with (
            patch(
                "kiro_crew.dashboard.chat_handlers.selectable_backends",
                return_value=frozenset({ACP_BACKEND_KIRO, ACP_BACKEND_CODEX}),
            ),
            patch(
                "kiro_crew.dashboard.chat_handlers.probe_backend",
                return_value=_installed(ACP_BACKEND_CODEX),
            ),
            patch(
                "kiro_crew.dashboard.chat_handlers._persist_sticky_backend",
                new=AsyncMock(),
            ),
        ):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat/slots/test/backend", json={"backend": ACP_BACKEND_CODEX}
                )
                assert resp.status == 200
                body = await resp.json()

        assert body["ok"] is True
        assert body["backend"] == ACP_BACKEND_CODEX
        assert slot.acp_backend == ACP_BACKEND_CODEX
        # A kiro model id means nothing to Codex, so the switch drops the pin.
        assert slot.model == ""
        assert body["model"] == ""
        # No live switch exists for a harness change — it always resets.
        state.sessions.reset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_switch_persists_the_sticky_default_for_new_sessions(self) -> None:
        slot = _ChatSlot("test")
        state = _mock_state(slot)
        sticky = AsyncMock()
        with (
            patch(
                "kiro_crew.dashboard.chat_handlers.selectable_backends",
                return_value=frozenset({ACP_BACKEND_KIRO, ACP_BACKEND_CODEX}),
            ),
            patch(
                "kiro_crew.dashboard.chat_handlers.probe_backend",
                return_value=_installed(ACP_BACKEND_CODEX),
            ),
            patch("kiro_crew.dashboard.chat_handlers._persist_sticky_backend", new=sticky),
        ):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat/slots/test/backend", json={"backend": ACP_BACKEND_CODEX}
                )
                assert resp.status == 200
        sticky.assert_awaited_once_with(ACP_BACKEND_CODEX)

    @pytest.mark.asyncio
    async def test_same_backend_is_a_no_op_and_does_not_reset(self) -> None:
        slot = _ChatSlot("test")
        slot.acp_backend = ACP_BACKEND_CODEX
        slot.model = "gpt-5.6-sol"
        state = _mock_state(slot)
        with (
            patch(
                "kiro_crew.dashboard.chat_handlers.selectable_backends",
                return_value=frozenset({ACP_BACKEND_KIRO, ACP_BACKEND_CODEX}),
            ),
            patch(
                "kiro_crew.dashboard.chat_handlers.probe_backend",
                return_value=_installed(ACP_BACKEND_CODEX),
            ),
        ):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat/slots/test/backend", json={"backend": ACP_BACKEND_CODEX}
                )
                assert resp.status == 200
        state.sessions.reset.assert_not_awaited()
        # A no-op must not drop a model the session is legitimately running.
        assert slot.model == "gpt-5.6-sol"

    @pytest.mark.asyncio
    async def test_unselectable_backend_is_refused_without_touching_the_slot(self) -> None:
        slot = _ChatSlot("test")
        state = _mock_state(slot)
        with patch(
            "kiro_crew.dashboard.chat_handlers.selectable_backends",
            return_value=frozenset({ACP_BACKEND_KIRO}),
        ):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat/slots/test/backend", json={"backend": ACP_BACKEND_CLAUDE}
                )
                assert resp.status == 400
                assert (await resp.json())["code"] == "invalid_backend"
        assert slot.acp_backend is None
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_adapter_answers_409_with_the_install_command(self) -> None:
        slot = _ChatSlot("test")
        state = _mock_state(slot)
        with (
            patch(
                "kiro_crew.dashboard.chat_handlers.selectable_backends",
                return_value=frozenset({ACP_BACKEND_KIRO, ACP_BACKEND_CODEX}),
            ),
            patch(
                "kiro_crew.dashboard.chat_handlers.probe_backend",
                return_value=_missing(ACP_BACKEND_CODEX),
            ),
        ):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat/slots/test/backend", json={"backend": ACP_BACKEND_CODEX}
                )
                assert resp.status == 409
                body = await resp.json()
        assert body["code"] == "backend_unavailable"
        assert body["install_command"] == "npm i -g pkg"
        assert slot.acp_backend is None

    @pytest.mark.asyncio
    async def test_unknown_slot_is_404(self) -> None:
        async with TestClient(TestServer(_make_app(_mock_state()))) as client:
            resp = await client.post("/api/chat/slots/nope/backend", json={"backend": ""})
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_non_string_backend_is_refused(self) -> None:
        slot = _ChatSlot("test")
        async with TestClient(TestServer(_make_app(_mock_state(slot)))) as client:
            resp = await client.post("/api/chat/slots/test/backend", json={"backend": 7})
            assert resp.status == 400
            assert (await resp.json())["code"] == "invalid_backend"


class TestModelsEndpointScopesToTheSlot:
    """``/api/models?slot=`` must answer for the session, not the global default."""

    @staticmethod
    def _request(slots: dict, query: dict) -> MagicMock:
        request = MagicMock()
        request.query = query
        state = MagicMock()
        state._slots = slots
        request.app = {"state": state}
        return request

    def test_no_slot_param_falls_back_to_the_configured_harness(self) -> None:
        req = self._request({}, {})
        assert _slot_backend(req, ACP_BACKEND_CLAUDE) == ACP_BACKEND_CLAUDE

    def test_slot_harness_wins_over_the_configured_one(self) -> None:
        slot = _ChatSlot("s")
        slot.acp_backend = ACP_BACKEND_CODEX
        req = self._request({"s": slot}, {"slot": "s"})
        # The global default is Claude, but this session runs Codex — answering
        # from the global value would offer models its adapter cannot serve.
        assert _slot_backend(req, ACP_BACKEND_CLAUDE) == ACP_BACKEND_CODEX

    def test_unknown_slot_falls_back_rather_than_erroring(self) -> None:
        req = self._request({}, {"slot": "gone"})
        assert _slot_backend(req, ACP_BACKEND_CLAUDE) == ACP_BACKEND_CLAUDE

    def test_an_explicit_kiro_pick_beats_the_configured_default(self) -> None:
        slot = _ChatSlot("s")
        slot.acp_backend = ACP_BACKEND_KIRO  # the user chose kiro-cli
        req = self._request({"s": slot}, {"slot": "s"})
        assert _slot_backend(req, ACP_BACKEND_CODEX) == ACP_BACKEND_KIRO

    def test_a_slot_that_never_picked_follows_the_configured_default(self) -> None:
        slot = _ChatSlot("s")  # acp_backend is None
        req = self._request({"s": slot}, {"slot": "s"})
        assert _slot_backend(req, ACP_BACKEND_CODEX) == ACP_BACKEND_CODEX


class TestPerBackendModelTranslation:
    """``acp_effective_model`` must answer in the harness the session will run in."""

    def test_own_catalog_backend_leaves_auto_unresolved(self) -> None:
        cfg = KiroCrewConfig()
        assert (
            cfg.acp_effective_model(None, None, global_model="auto", backend=ACP_BACKEND_CODEX)
            == ""
        )

    def test_kiro_resolves_auto_through_its_own_spec(self) -> None:
        cfg = KiroCrewConfig()
        kiro = cfg.acp_effective_model(
            None, "opus-4.8", global_model="auto", backend=ACP_BACKEND_KIRO
        )
        claude = cfg.acp_effective_model(
            None, "opus-4.8", global_model="auto", backend=ACP_BACKEND_CLAUDE
        )
        # Same canonical key, two namespaces — which is exactly why the switch
        # clears the slot's pin instead of carrying it across.
        assert kiro and claude and kiro != claude

    def test_backend_defaults_to_the_configured_one(self) -> None:
        cfg = KiroCrewConfig()
        cfg.agent.acp_backend = ACP_BACKEND_CODEX
        assert cfg.acp_effective_model(None, None, global_model="auto") == ""

    def test_every_external_adapter_owns_its_catalog(self) -> None:
        # All three advertise the ids they accept and take them verbatim, so
        # none of them may resolve ``auto`` through kiro-cli's agent spec.
        # Claude's membership is the fix for the reported crash: that spec is
        # ``~/.kiro/agents/kirocrew.json``, and a GPT model pinned there for
        # kiro-cli was handed to claude-agent-acp, which refused it.
        assert ACP_BACKEND_CLAUDE in ACP_BACKENDS_OWN_MODEL_CATALOG
        assert ACP_BACKEND_CODEX in ACP_BACKENDS_OWN_MODEL_CATALOG
        # Kiro-cli is the harness that spec belongs to, so it must stay out.
        assert ACP_BACKEND_KIRO not in ACP_BACKENDS_OWN_MODEL_CATALOG

    def test_a_kiro_agent_spec_model_never_reaches_the_claude_adapter(self, tmp_path) -> None:
        """The reported ``-32603``, reproduced through the resolver that caused it.

        ``agent.model`` is ``auto`` (the shipped default), so the collapsed
        global came from the installed kiro agent spec -- whose model is
        whatever the operator runs kiro-cli on. On a GPT pin that id went
        straight to claude-agent-acp.
        """
        import json

        from kiro_crew.config import loader as loader_mod

        spec = tmp_path / "kirocrew.json"
        spec.write_text(json.dumps({"name": "kirocrew", "model": "gpt-5.6-sol"}))

        cfg = KiroCrewConfig()
        cfg.agent.model = "auto"
        # ``global_model`` is left unset on purpose: passing one skips the very
        # collapse under test, which is where the spec is read.
        with patch.object(loader_mod, "kiro_agents_dir", return_value=tmp_path):
            # kiro-cli still resolves through its own spec: that is its file.
            assert cfg.acp_effective_model(None, None, backend=ACP_BACKEND_KIRO) == "gpt-5.6-sol"
            # The adapters do not read it at all.
            for backend in (ACP_BACKEND_CLAUDE, ACP_BACKEND_CODEX):
                assert cfg.acp_effective_model(None, None, backend=backend) == ""
