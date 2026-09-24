"""Model and effort pickers read an adapted harness's catalog before any session.

The setup case: an operator opening Settings to choose a harness, its default
model and its effort level has by definition never opened a chat on it. An
adapted harness reports its models and effort ladder only in ``session/new``, so
without an offline read the picker offers a static snapshot that misses every
model newer than it (or only the ``auto`` sentinel).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.acp_backends import (
    ACP_BACKEND_CLAUDE,
    ACP_BACKEND_CODEX,
    ACP_BACKEND_KIRO,
    ACP_BACKEND_OPENCODE,
)
from kiro_crew.agent_sdk.capabilities import capabilities_for


class _FakeProvider:
    def __init__(self, models: list[dict], backend: str) -> None:
        self._models = models
        self.capabilities = capabilities_for(backend)
        self.client = SimpleNamespace(backend=backend)

    def available_models(self) -> list[dict]:
        return self._models


def _request(providers=(), slots=None):
    sessions = SimpleNamespace(
        active_providers=lambda: list(providers), get_provider=lambda _k: None
    )
    state = SimpleNamespace(sessions=sessions, _slots=slots or {})
    req = MagicMock()
    req.app.__getitem__.return_value = state
    req.app.get = lambda key, default=None: state if key == "state" else default
    req.query = {}
    return req


def _catalog(models, levels=()):
    from kiro_crew.acp.adapter_catalog import AdapterCatalog

    return AdapterCatalog(
        models=[{"modelId": m, "name": m.upper()} for m in models],
        effort_levels=list(levels),
    )


@pytest.fixture
def cfg(monkeypatch):
    from kiro_crew.dashboard.handlers import agents

    holder = SimpleNamespace(
        agent=SimpleNamespace(acp_backend=ACP_BACKEND_KIRO, model="", sandbox="auto")
    )
    monkeypatch.setattr(agents.KiroCrewConfig, "load", staticmethod(lambda: holder))
    return holder


class TestColdPickersReadTheAdapterOffline:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "backend,probed",
        [
            (ACP_BACKEND_CLAUDE, ["opus", "claude-fable-5-1[1m]"]),
            (ACP_BACKEND_CODEX, ["gpt-6-sol", "gpt-6-luna"]),
            (ACP_BACKEND_OPENCODE, ["openai/gpt-6-sol"]),
        ],
    )
    async def test_a_cold_picker_offers_the_probed_catalog(self, cfg, backend, probed, monkeypatch):
        from kiro_crew.dashboard.handlers import agents

        monkeypatch.setattr(
            agents, "_probe_adapter_catalog", AsyncMock(return_value=_catalog(probed))
        )
        request = _request()
        request.query = {"backend": backend}

        names = [r["model_name"] for r in json.loads((await agents.api_models(request)).body)]

        # Auto leads; every probed id is offered VERBATIM -- the values the
        # adapter's own set_config_option accepts.
        assert names[0] == "auto"
        assert set(probed) <= set(names)

    @pytest.mark.asyncio
    async def test_the_probed_adapter_label_is_shown(self, cfg, monkeypatch):
        # The adapter names the model its alias serves today ("Opus 5.5"); a
        # registry label for the same alias would name an older model.
        from kiro_crew.acp.adapter_catalog import AdapterCatalog
        from kiro_crew.dashboard.handlers import agents

        catalog = AdapterCatalog(models=[{"modelId": "opus", "name": "Opus 5.5"}])
        monkeypatch.setattr(agents, "_probe_adapter_catalog", AsyncMock(return_value=catalog))
        request = _request()
        request.query = {"backend": ACP_BACKEND_CODEX}

        rows = json.loads((await agents.api_models(request)).body)

        assert {"model_name": "opus", "display_name": "Opus 5.5"}.items() <= rows[1].items()

    @pytest.mark.asyncio
    async def test_a_live_session_wins_and_no_probe_is_spawned(self, cfg, monkeypatch):
        from kiro_crew.dashboard.handlers import agents

        probe = AsyncMock(return_value=_catalog(["never-used"]))
        monkeypatch.setattr(agents, "_probe_adapter_catalog", probe)
        live = _FakeProvider([{"modelId": "gpt-6-sol", "name": "6 Sol"}], ACP_BACKEND_CODEX)
        request = _request([live])
        request.query = {"backend": ACP_BACKEND_CODEX}

        names = [r["model_name"] for r in json.loads((await agents.api_models(request)).body)]

        assert "gpt-6-sol" in names and "never-used" not in names
        probe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_slot_on_its_own_harness_is_answered_for_that_harness(self, cfg, monkeypatch):
        from kiro_crew.dashboard.handlers import agents

        probe = AsyncMock(return_value=_catalog(["gpt-6-sol"]))
        monkeypatch.setattr(agents, "_probe_adapter_catalog", probe)
        request = _request(slots={"s1": SimpleNamespace(acp_backend=ACP_BACKEND_CODEX)})
        request.query = {"slot": "s1"}

        names = [r["model_name"] for r in json.loads((await agents.api_models(request)).body)]

        assert "gpt-6-sol" in names
        assert probe.await_args.args[1] == ACP_BACKEND_CODEX

    def test_an_unset_slot_harness_inherits_and_empty_means_kiro(self) -> None:
        from kiro_crew.dashboard.handlers import agents

        slots = {
            "inherit": SimpleNamespace(acp_backend=None),
            "kiro": SimpleNamespace(acp_backend=""),
        }
        req = _request(slots=slots)
        req.query = {"slot": "inherit"}
        assert agents._slot_backend(req, ACP_BACKEND_CODEX) == ACP_BACKEND_CODEX
        req.query = {"slot": "kiro"}
        assert agents._slot_backend(req, ACP_BACKEND_CODEX) == ACP_BACKEND_KIRO

    def test_an_unservable_backend_query_degrades_to_kiro(self) -> None:
        from kiro_crew.dashboard.handlers import agents

        req = _request()
        req.query = {"backend": "no-such-harness"}
        assert agents._slot_backend(req, ACP_BACKEND_CODEX) == ACP_BACKEND_KIRO

    @pytest.mark.asyncio
    async def test_effort_levels_answer_the_named_harnesss_own_ladder(self, cfg, monkeypatch):
        from kiro_crew.dashboard.handlers import agents

        monkeypatch.setattr(
            agents,
            "_probe_adapter_catalog",
            AsyncMock(return_value=_catalog([], ["low", "medium", "max"])),
        )
        request = _request()
        request.query = {"backend": ACP_BACKEND_CODEX}

        assert json.loads((await agents.api_effort_levels(request)).body) == [
            "low",
            "medium",
            "max",
        ]

    @pytest.mark.asyncio
    async def test_effort_levels_for_kiro_never_probe(self, cfg, monkeypatch):
        from kiro_crew.dashboard.handlers import agents

        probe = AsyncMock()
        monkeypatch.setattr(agents, "_probe_adapter_catalog", probe)
        request = _request()
        request.query = {"backend": ""}

        await agents.api_effort_levels(request)

        probe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failed_probe_answers_200_with_auto_first(self, cfg, monkeypatch):
        """Not installed / not signed in must not 5xx a read-only picker."""
        from kiro_crew.acp.adapter_catalog import AdapterCatalog
        from kiro_crew.dashboard.handlers import agents

        monkeypatch.setattr(
            agents, "_probe_adapter_catalog", AsyncMock(return_value=AdapterCatalog())
        )
        for backend in (ACP_BACKEND_CLAUDE, ACP_BACKEND_CODEX):
            request = _request()
            request.query = {"backend": backend}
            response = await agents.api_models(request)
            assert response.status == 200
            assert json.loads(response.body)[0]["model_name"] == "auto"


class TestAdapterCatalogProbe:
    """The probe spawns an adapter, so it must spawn as rarely as possible."""

    @pytest.fixture(autouse=True)
    def _clean_cache(self):
        from kiro_crew.acp import adapter_catalog

        adapter_catalog.invalidate()
        yield
        adapter_catalog.invalidate()

    @pytest.mark.asyncio
    async def test_a_success_is_cached_and_concurrent_readers_share_one_spawn(
        self, tmp_path, monkeypatch
    ):
        from kiro_crew.acp import adapter_catalog

        calls = 0

        async def _handshake(backend, work_dir, sandbox_mode):
            nonlocal calls
            calls += 1
            # Yield, so a second reader is genuinely in flight when this returns
            # -- a single-flight that only works serially is not one.
            await asyncio.sleep(0)
            return adapter_catalog.AdapterCatalog(models=[{"modelId": "opus"}])

        monkeypatch.setattr(adapter_catalog, "_handshake", _handshake)

        first = await asyncio.gather(
            *[adapter_catalog.read_catalog(ACP_BACKEND_CLAUDE, tmp_path, "auto") for _ in range(5)]
        )
        again = await adapter_catalog.read_catalog(ACP_BACKEND_CLAUDE, tmp_path, "auto")

        assert calls == 1, "concurrent readers each spawned an adapter"
        assert all(c.models for c in first) and again.models

    @pytest.mark.asyncio
    async def test_a_failure_never_escapes_and_is_retried_sooner_than_a_success(
        self, tmp_path, monkeypatch
    ):
        """The common failures are the ones an operator fixes and retries.

        Holding a failure for the success TTL would read as the fix not having
        worked; holding it for nothing would turn a degraded picker's poll loop
        into a spawn loop.
        """
        from kiro_crew.acp import adapter_catalog

        async def _boom(backend, work_dir, sandbox_mode):
            raise RuntimeError("adapter not installed")

        monkeypatch.setattr(adapter_catalog, "_handshake", _boom)

        catalog = await adapter_catalog.read_catalog(ACP_BACKEND_CLAUDE, tmp_path, "auto")

        assert not catalog.models and not catalog.effort_levels
        assert adapter_catalog.FAILURE_TTL_SECONDS < adapter_catalog.CACHE_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_a_harness_that_answers_offline_is_never_probed(self, tmp_path, monkeypatch):
        """kiro-cli and KAS have their own catalog command; spawning an ACP
        session to re-learn it would be pure cost."""
        from kiro_crew.acp import adapter_catalog

        async def _unexpected(backend, work_dir, sandbox_mode):
            raise AssertionError(f"probed {backend!r}, which answers offline")

        monkeypatch.setattr(adapter_catalog, "_handshake", _unexpected)

        assert not await adapter_catalog.read_catalog("", tmp_path, "auto")
        assert not await adapter_catalog.read_catalog("kas", tmp_path, "auto")


class TestClaudeLabels:
    def test_the_adapters_label_wins_over_a_folded_registry_row(self) -> None:
        from kiro_crew.dashboard.handlers import agents

        live = _FakeProvider([{"modelId": "opus", "name": "Opus 5.5"}], ACP_BACKEND_CLAUDE)
        rows = agents._cc_models(_request([live]))
        opus = next(r for r in rows if r["model_name"] == "opus")

        assert opus["display_name"] == "Opus 5.5"
        assert opus["context_window"]

    def test_a_missing_adapter_label_keeps_the_registrys(self) -> None:
        from kiro_crew.dashboard.handlers import agents

        live = _FakeProvider([{"modelId": "opus", "name": "opus"}], ACP_BACKEND_CLAUDE)
        opus = next(r for r in agents._cc_models(_request([live])) if r["model_name"] == "opus")

        assert opus["display_name"] != "opus"


class TestLiveSwitchWireId:
    """A live model switch sends the adapter the id it advertised."""

    @staticmethod
    def _provider(backend: str, models: list[str]):
        from kiro_crew.providers.acp import AcpProvider

        provider = MagicMock(spec=AcpProvider)
        provider.capabilities = capabilities_for(backend)
        provider.client = SimpleNamespace(backend=backend)
        provider.available_models = lambda: [{"modelId": m} for m in models]
        return provider

    def test_an_advertised_claude_alias_goes_out_verbatim(self) -> None:
        from kiro_crew.dashboard.chat_handlers import _wire_model_id

        provider = self._provider(ACP_BACKEND_CLAUDE, ["default", "opus", "sonnet"])
        assert _wire_model_id(provider, "opus") == "opus"

    def test_an_unadvertised_canonical_key_is_still_translated(self) -> None:
        from kiro_crew import model_registry
        from kiro_crew.dashboard.chat_handlers import _wire_model_id

        provider = self._provider(ACP_BACKEND_CLAUDE, ["opus"])
        assert _wire_model_id(provider, "opus-4.8-1m") == model_registry.to_provider_id(
            "opus-4.8-1m", "claude_code"
        )

    def test_auto_still_needs_a_reset(self) -> None:
        from kiro_crew.dashboard.chat_handlers import _wire_model_id

        assert _wire_model_id(self._provider(ACP_BACKEND_CLAUDE, ["opus"]), "auto") == ""


@pytest.mark.asyncio
async def test_a_runtime_driven_harness_is_never_probed(tmp_path, monkeypatch) -> None:
    """Codex runs on the multiplexed runtime, whose handshake this client cannot
    speak; probing it would spawn an adapter only to have it refused."""
    from kiro_crew.acp import adapter_catalog
    from kiro_crew.acp_backends import ACP_BACKENDS_ACP_RUNTIME

    async def _unexpected(backend, work_dir, sandbox_mode):
        raise AssertionError(f"probed {backend!r}, which the runtime drives")

    monkeypatch.setattr(adapter_catalog, "_handshake", _unexpected)
    adapter_catalog.invalidate()
    for backend in ACP_BACKENDS_ACP_RUNTIME:
        assert not await adapter_catalog.read_catalog(backend, tmp_path, "auto")
    assert ACP_BACKEND_CLAUDE in adapter_catalog.PROBE_BACKENDS
