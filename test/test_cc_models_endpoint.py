"""Tests for the adapter-backed model lists assembled by ``/api/models``.

Concrete choices come from a live ACP session. A cold adapter exposes only its
``auto`` sentinel until it reports the account's advertised set.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.acp_backends import ACP_BACKEND_CLAUDE, ACP_BACKEND_CODEX
from kiro_crew.dashboard.handlers.agents import (
    _advertised_cc_models,
    _advertised_codex_models,
    _cc_models,
    _codex_models,
    _normalize_model_key,
)


def _request_with_providers(providers: dict) -> MagicMock:
    """Fake aiohttp request whose sessions.active_providers() yields `providers`.

    Mirrors the real SessionManager API (active_providers()) so the test can't
    pass against an attribute the production object doesn't have.
    """
    sessions = SimpleNamespace(active_providers=lambda: list(providers.values()))
    state = SimpleNamespace(sessions=sessions)
    req = MagicMock()
    req.app.__getitem__.return_value = state
    return req


class _FakeProvider:
    def __init__(self, models, backend: str = "claude"):
        self._models = models
        self.client = SimpleNamespace(backend=backend)

    def available_models(self):
        return self._models


class TestAdvertisedCcModels:
    def test_maps_modelid_name_description(self):
        # An unknown provider id (not in the registry) passes through unchanged.
        prov = _FakeProvider(
            [
                {"modelId": "claude-sonnet-4-6", "name": "Sonnet 4.6", "description": "Everyday"},
            ]
        )
        out = _advertised_cc_models(_request_with_providers({"s": prov}))
        assert out == [
            {
                "model_name": "claude-sonnet-4-6",
                "display_name": "Sonnet 4.6",
                "description": "Everyday",
            }
        ]

    def test_a_registry_known_id_is_still_passed_through_verbatim(self):
        # Even an id the registry can canonicalize stays as advertised: the
        # adapter accepts what it offered, and only that. Folding it to
        # ``opus-4.8-1m`` would put a value in the picker that
        # ``session/set_config_option`` refuses. Dedup still folds it -- that
        # happens in ``_cc_models`` via ``_normalize_model_key``, on a copy.
        prov = _FakeProvider(
            [
                {
                    "modelId": "global.anthropic.claude-opus-4-8[1m]",
                    "name": "Opus 4.8",
                    "description": "",
                },
            ]
        )
        out = _advertised_cc_models(_request_with_providers({"s": prov}))
        assert out[0]["model_name"] == "global.anthropic.claude-opus-4-8[1m]"

    def test_empty_when_no_active_sessions(self):
        assert _advertised_cc_models(_request_with_providers({})) == []

    def test_skips_provider_without_accessor(self):
        out = _advertised_cc_models(_request_with_providers({"s": object()}))
        assert out == []


class TestCodexModels:
    def test_cold_picker_offers_only_auto(self):
        out = _codex_models(_request_with_providers({}))
        assert [row["model_name"] for row in out] == ["auto"]

    def test_advertised_models_keep_codex_wire_ids(self):
        provider = _FakeProvider(
            [
                {
                    "modelId": "codex-model-from-adapter",
                    "name": "Adapter model",
                    "description": "Served to this account",
                }
            ],
            ACP_BACKEND_CODEX,
        )

        advertised = _advertised_codex_models(_request_with_providers({"s": provider}))
        assert advertised[0]["model_name"] == "codex-model-from-adapter"

        rows = _codex_models(_request_with_providers({"s": provider}))
        assert [row["model_name"] for row in rows] == [
            "auto",
            "codex-model-from-adapter",
        ]


class TestCcModelsMerge:
    def test_cold_picker_offers_only_auto(self):
        # No live provider means entitlement is unknown. Guessing concrete ids
        # would violate the advertised-model contract and fail for some accounts.
        out = _cc_models(_request_with_providers({}))
        names = [m["model_name"] for m in out]
        assert names == ["auto"]

    def test_advertised_set_filters_the_registry(self):
        """The advertised set is authoritative: unentitled registry rows go away.

        This is the free-tier case. Previously the registry led unconditionally and
        the adapter could only ADD, so an account served two models was still
        offered the full flagship list and only found out at prompt time.
        """
        prov = _FakeProvider(
            [{"modelId": "global.anthropic.claude-sonnet-4-6[1m]", "name": "Sonnet 4.6"}]
        )
        out = _cc_models(_request_with_providers({"s": prov}))
        names = [m["model_name"] for m in out]
        assert names[0] == "auto"
        assert "global.anthropic.claude-sonnet-4-6[1m]" in names
        # The flagship is in the registry but was NOT advertised → filtered out.
        assert "opus-4.8-1m" not in names
        assert "opus-4.8" not in names

    def test_the_adapters_own_label_wins_over_the_registrys(self):
        """The live session describes the model; the registry file can be stale.

        An alias id folds onto whichever registry entry it named when that file
        was written, so the registry answers "Opus 4.8 (1M context)" for an
        ``opus`` alias an account now serves as Opus 5 -- a label the operator
        would read as the model they are running. The registry is still consulted
        for the context window, which the adapter does not report.
        """
        prov = _FakeProvider([{"modelId": "opus", "name": "Opus", "description": "Opus 5"}])
        out = _cc_models(_request_with_providers({"s": prov}))
        row = next(m for m in out if m["model_name"] == "opus")
        assert row["display_name"] == "Opus"
        assert row["description"] == "Opus 5"
        assert row["context_window"] == 1_000_000

    def test_the_registry_label_fills_in_when_the_adapter_supplies_none(self):
        prov = _FakeProvider(
            [{"modelId": "global.anthropic.claude-sonnet-4-6[1m]", "name": "", "description": ""}]
        )
        out = _cc_models(_request_with_providers({"s": prov}))
        row = next(m for m in out if m["model_name"] == "global.anthropic.claude-sonnet-4-6[1m]")
        assert row["display_name"] == "Sonnet 4.6 (1M context)"

    def test_unknown_advertised_models_still_pass_through(self):
        # Forward-compat: a model the registry does not list is still offered when
        # the backend advertises it, otherwise a newly-served model is unreachable.
        prov = _FakeProvider(
            [
                {"modelId": "claude-opus-4-1", "name": "Opus 4.1", "description": ""},
                {"modelId": "claude-sonnet-4-5", "name": "Sonnet 4.5", "description": ""},
            ]
        )
        out = _cc_models(_request_with_providers({"s": prov}))
        names = [m["model_name"] for m in out]
        assert "claude-opus-4-1" in names
        assert "claude-sonnet-4-5" in names
        # And the unentitled registry flagship is gone.
        assert "opus-4.8-1m" not in names
        # "auto" still leads and is never filtered by entitlement -- it is the
        # configured-default sentinel, not a model the backend serves.
        assert names[0] == "auto"

    def test_configured_default_is_not_resurrected_when_unentitled(self):
        """A stale config pick must not outlive the entitlement.

        Force-including it would reintroduce exactly the unusable option the
        filter removes.
        """
        prov = _FakeProvider(
            [{"modelId": "global.anthropic.claude-sonnet-4-6[1m]", "name": "Sonnet 4.6"}]
        )
        out = _cc_models(_request_with_providers({"s": prov}), configured_default="opus-4.8-1m")
        names = [m["model_name"] for m in out]
        assert "opus-4.8-1m" not in names
        assert names[0] == "auto"

    def test_configured_default_is_not_invented_when_nothing_advertised(self):
        # A persisted value is not an entitlement advertisement.
        out = _cc_models(_request_with_providers({}), configured_default="some-custom-model")
        names = [m["model_name"] for m in out]
        assert names == ["auto"]

    def test_no_duplicate_when_adapter_lists_known_model(self):
        # The adapter advertises provider ids that ARE in the registry; mapped
        # back to canonical keys they collapse to one row each (registry wins).
        prov = _FakeProvider(
            [
                {
                    "modelId": "global.anthropic.claude-sonnet-4-6[1m]",
                    "name": "Sonnet 4.6",
                    "description": "",
                },
                {
                    "modelId": "global.anthropic.claude-opus-4-8[1m]",
                    "name": "Opus 4.8",
                    "description": "",
                },
            ]
        )
        out = _cc_models(_request_with_providers({"s": prov}))
        names = [m["model_name"] for m in out]
        assert names.count("global.anthropic.claude-opus-4-8[1m]") == 1
        assert names.count("global.anthropic.claude-sonnet-4-6[1m]") == 1

    def test_a_known_id_still_picks_up_the_registrys_context_window(self):
        # The registry contributes what the adapter does not report. The wire
        # value and the label both stay the adapter's.
        prov = _FakeProvider(
            [
                {
                    "modelId": "global.anthropic.claude-opus-4-8[1m]",
                    "name": "Opus 4.8",
                    "description": "",
                },
            ]
        )
        out = _cc_models(_request_with_providers({"s": prov}))
        opus48 = next(m for m in out if m["model_name"] == "global.anthropic.claude-opus-4-8[1m]")
        assert opus48["display_name"] == "Opus 4.8"
        assert opus48["context_window"] == 1_000_000

    def test_configured_default_force_included(self):
        out = _cc_models(_request_with_providers({}), configured_default="custom-model-xyz")
        names = [m["model_name"] for m in out]
        assert names == ["auto"]

    def test_configured_default_not_duplicated_if_already_present(self):
        out = _cc_models(
            _request_with_providers({}),
            configured_default="opus-4.8-1m",
        )
        names = [m["model_name"] for m in out]
        assert names == ["auto"]

    def test_configured_default_auto_does_not_insert_blank_row(self):
        # cc_model="auto" round-trips to "" (auto's provider id is empty), which
        # must NOT be inserted as a blank-named row at the top of the dropdown —
        # the "auto" registry row already covers it.
        out = _cc_models(_request_with_providers({}), configured_default="auto")
        names = [m["model_name"] for m in out]
        assert "" not in names
        assert all(m["model_name"] for m in out)
        # the canonical "auto" row is still present, exactly once.
        assert names.count("auto") == 1


class TestNormalizeModelKey:
    """`_normalize_model_key` routes through the canonical registry (#5339).

    Mirror of the frontend `normalizeModelKey` unit tests in
    `website/src/test/model.displayModel.test.ts` -- the two must agree, which is
    the whole point of folding through the shared `model_registry.json`.
    """

    def test_auto_default_and_unset(self):
        # auto/default fold to the sentinel; an unset id stays "" (distinct).
        assert _normalize_model_key(" auto ") == "auto"
        assert _normalize_model_key("default") == "auto"
        assert _normalize_model_key("DEFAULT") == "auto"
        assert _normalize_model_key("") == ""
        assert _normalize_model_key("   ") == ""

    def test_alias_key_and_provider_id_fold_to_one_key(self):
        # An alias, the canonical key, and the claude_code provider id (with or
        # without a routing prefix) all resolve to one canonical key, any case.
        assert _normalize_model_key("claude-opus-4.8") == "opus-4.8-1m"
        assert _normalize_model_key("Claude-Opus-4.8") == "opus-4.8-1m"
        assert _normalize_model_key("opus-4.8-1m") == "opus-4.8-1m"
        assert _normalize_model_key("opus") == "opus-4.8-1m"
        assert _normalize_model_key("global.anthropic.claude-opus-4-8[1m]") == "opus-4.8-1m"
        # The "fold a provider/partition prefix" half of #5339: a regional
        # profile id that is not itself a registry entry folds after the peel.
        assert _normalize_model_key("us.anthropic.claude-opus-4-8[1m]") == "opus-4.8-1m"

    def test_distinct_context_window_variants_stay_apart(self):
        # The old dot->dash fold made both of these `claude-opus-4-8`, equating a
        # 200K model with a 1M one. The registry lists them as separate entries.
        assert _normalize_model_key("claude-opus-4-8") == "opus-4.8"  # 200K
        assert _normalize_model_key("claude-opus-4.8") == "opus-4.8-1m"  # 1M
        assert _normalize_model_key("claude-opus-4-8") != _normalize_model_key("claude-opus-4.8")

    def test_kiro_distinct_models_stay_apart_via_acp_first_fold(self):
        # The claude_code index aliases these onto Sonnet/Opus 4.8 for dropdown
        # dedup, but kiro serves them as DISTINCT real models. Resolving the acp
        # index first (canonical_key's documented order) keeps them apart, so the
        # shared fold cannot equate a Haiku pin with Sonnet 4.6 (a real 1M->200K
        # swap the downgrade flag must catch).
        assert _normalize_model_key("claude-haiku-4.5") == "haiku-4.5"
        assert _normalize_model_key("claude-sonnet-4.5") == "sonnet-4.5"
        assert _normalize_model_key("claude-sonnet-4") == "sonnet-4"
        assert _normalize_model_key("claude-opus-4.6") == "opus-4.6-1m"
        assert _normalize_model_key("claude-sonnet-4.6") == "sonnet-4.6-1m"
        assert _normalize_model_key("claude-haiku-4.5") != _normalize_model_key("claude-sonnet-4.6")
        assert _normalize_model_key("claude-opus-4.6") != _normalize_model_key("claude-opus-4.8")
        # acp-only canonical keys resolve to themselves.
        assert _normalize_model_key("haiku-4.5") == "haiku-4.5"
        assert _normalize_model_key("opus-4.6-1m") == "opus-4.6-1m"

    def test_unregistered_id_uses_the_string_fold(self):
        # GPT/DeepSeek/Qwen and future models are absent from the (Anthropic-only)
        # registry, so they keep the historical trim/lowercase/dot->dash fold.
        assert _normalize_model_key("GPT-5.6") == "gpt-5-6"
        assert _normalize_model_key("deepseek-3.2") == "deepseek-3-2"
        assert _normalize_model_key("claude-opus-5") == "claude-opus-5"


class TestPickerNeverCrossesBackends:
    """A picker for one backend must show ONLY that backend's models.

    A live Claude session and a live Codex session can coexist (two different
    chat slots), and each backend's own `/api/models?backend=` request must
    see only its own advertised set — never the other's rows leaking in
    because both providers sit in the same `active_providers()` list.
    """

    def test_claude_request_never_sees_a_live_codex_providers_models(self):
        claude_provider = _FakeProvider(
            [{"modelId": "claude-opus-4-8", "name": "Opus 4.8", "description": ""}],
            "claude",
        )
        codex_provider = _FakeProvider(
            [{"modelId": "gpt-5.6-sol", "name": "GPT-5.6 Sol", "description": ""}],
            ACP_BACKEND_CODEX,
        )
        # Codex is the NEWER session (later in the dict / "active" order), so a
        # filter that merely reads the first or last provider rather than
        # filtering by backend would return the wrong list.
        request = _request_with_providers(
            {"claude-slot": claude_provider, "codex-slot": codex_provider}
        )

        cc_rows = [row["model_name"] for row in _cc_models(request)]
        codex_rows = [row["model_name"] for row in _codex_models(request)]

        assert any("opus" in name for name in cc_rows)
        assert not any("gpt" in name.lower() for name in cc_rows)
        assert "gpt-5.6-sol" in codex_rows
        assert not any("opus" in name for name in codex_rows)

    def test_codex_request_never_sees_a_live_claude_providers_models(self):
        claude_provider = _FakeProvider(
            [{"modelId": "claude-sonnet-4-6", "name": "Sonnet 4.6", "description": ""}],
            "claude",
        )
        codex_provider = _FakeProvider(
            [{"modelId": "gpt-5.6-luna", "name": "GPT-5.6 Luna", "description": ""}],
            ACP_BACKEND_CODEX,
        )
        request = _request_with_providers(
            {"codex-slot": codex_provider, "claude-slot": claude_provider}
        )

        codex_rows = [row["model_name"] for row in _codex_models(request)]

        assert codex_rows == ["auto", "gpt-5.6-luna"]


class TestClaudePickerPopulatesFromTheAdaptersOwnCatalog:
    """Reproduces the exact report: pick Claude, picker shows only "Auto".

    The picker renders what a live session advertised, and claude-agent-acp
    advertises its catalog in ``configOptions[id="model"]`` -- its
    ``session/new`` result carries no ``models`` block at all. Reading only that
    block left the list permanently empty however healthy the session was.

    The payload below is the shape the installed adapter really returns, so this
    covers the report end to end through the real client rather than through a
    hand-built provider stub.
    """

    @staticmethod
    def _session_new() -> dict:
        return {
            "sessionId": "session",
            "configOptions": [
                {
                    "id": "model",
                    "name": "Model",
                    "type": "select",
                    "currentValue": "opus",
                    "options": [
                        {"value": "default", "name": "Default (recommended)"},
                        {"value": "sonnet", "name": "Sonnet", "description": "Sonnet 5"},
                        {"value": "opus", "name": "Opus", "description": "Opus 5"},
                        {"value": "haiku", "name": "Haiku", "description": "Haiku 4.5"},
                    ],
                }
            ],
        }

    def _live_claude_provider(self, tmp_path, model: str = "auto"):
        from kiro_crew.acp.client import AcpClient
        from kiro_crew.acp.types import ACP_BACKEND_CLAUDE
        from kiro_crew.providers.acp import AcpProvider

        client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_CLAUDE, model=model)
        client._session_id = "session"
        session_new = self._session_new()
        client._capture_available_models(session_new)
        client._store_session_config(session_new)
        provider = AcpProvider(work_dir=tmp_path, acp_backend=ACP_BACKEND_CLAUDE)
        provider._client = client
        return provider

    def test_a_live_session_reports_the_real_claude_catalog(self, tmp_path):
        provider = self._live_claude_provider(tmp_path)
        rows = _cc_models(_request_with_providers({"s": provider}))
        names = [row["model_name"] for row in rows]

        # "default" folds onto the Auto sentinel, leaving Auto + three models.
        assert names == ["auto", "sonnet", "opus", "haiku"]
        # Every id is one ``session/set_config_option`` accepts, so a pick from
        # this list can actually be applied.
        advertised = {m["modelId"] for m in provider.available_models()}
        assert set(names) - {"auto"} <= advertised

    def test_a_gpt_pin_from_the_kiro_agent_spec_never_empties_the_catalog(self, tmp_path):
        """The reported failure mode: an unservable pin must not cost the list.

        The catalog is captured at ``session/new``, before any model is applied,
        so it survives whatever the pin does -- and the pin is now refused
        against that same catalog rather than by the adapter mid-startup.
        """
        import asyncio

        from kiro_crew.acp.client import AcpModelUnavailable

        provider = self._live_claude_provider(tmp_path, model="gpt-5.6-sol")

        with pytest.raises(AcpModelUnavailable):
            asyncio.run(provider._client._apply_startup_model())

        names = [row["model_name"] for row in _cc_models(_request_with_providers({"s": provider}))]
        assert names == ["auto", "sonnet", "opus", "haiku"]
        assert not any("gpt" in name.lower() for name in names)


class TestColdPickersReadTheAdapterOffline:
    """A picker must offer real models BEFORE any session exists.

    That is the setup case, not an edge case: an operator opening Settings to
    choose a harness, its default model and its effort level has by definition
    never opened a chat on it. kiro-cli and OpenCode already answered offline
    (``chat --list-models`` / ``opencode models``); Claude and Codex published
    nothing but their ``auto`` sentinel until a session happened to be running,
    so both controls were empty exactly while being configured.
    """

    @staticmethod
    def _request(providers=()):
        sessions = SimpleNamespace(
            active_providers=lambda: list(providers), get_provider=lambda _k: None
        )
        state = SimpleNamespace(sessions=sessions, _slots={})
        req = MagicMock()
        req.app.__getitem__.return_value = state
        req.app.get = lambda key, default=None: state if key == "state" else default
        return req

    @staticmethod
    def _catalog(models, levels=()):
        from kiro_crew.acp.adapter_catalog import AdapterCatalog

        return AdapterCatalog(
            models=[{"modelId": m, "name": m.upper()} for m in models],
            effort_levels=list(levels),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "backend,probed",
        [
            (ACP_BACKEND_CLAUDE, ["sonnet", "opus"]),
            (ACP_BACKEND_CODEX, ["gpt-5.6-sol", "gpt-5.5"]),
        ],
    )
    async def test_a_cold_picker_offers_the_probed_catalog(self, backend, probed, monkeypatch):
        from kiro_crew.dashboard.handlers import agents

        monkeypatch.setattr(
            agents,
            "_probe_adapter_catalog",
            AsyncMock(return_value=self._catalog(probed, ["low", "high"])),
        )
        request = self._request()
        request.query = {"backend": backend}

        rows = json.loads((await agents.api_models(request)).body.decode())
        names = [row["model_name"] for row in rows]

        # Auto still leads, and every probed id is offered VERBATIM -- these are
        # the values the adapter's own set_config_option accepts.
        assert names[0] == "auto"
        assert set(probed) <= set(names)

    @pytest.mark.asyncio
    async def test_a_live_session_wins_and_no_probe_is_spawned(self, monkeypatch):
        """A running session is the authoritative answer; probing anyway would
        spawn a second adapter to re-learn what the live one already reported."""
        from kiro_crew.dashboard.handlers import agents

        probe = AsyncMock(return_value=self._catalog(["never-used"]))
        monkeypatch.setattr(agents, "_probe_adapter_catalog", probe)
        provider = _FakeProvider(
            [{"modelId": "opus", "name": "Opus", "description": ""}], ACP_BACKEND_CLAUDE
        )
        request = self._request([provider])
        request.query = {"backend": ACP_BACKEND_CLAUDE}

        names = [
            row["model_name"]
            for row in json.loads((await agents.api_models(request)).body.decode())
        ]

        assert "opus" in names
        assert "never-used" not in names
        probe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_effort_levels_answer_the_named_harnesss_own_ladder(self, monkeypatch):
        """Ladders differ per harness, so a static list offers refused levels."""
        from kiro_crew.dashboard.handlers import agents

        monkeypatch.setattr(
            agents,
            "_probe_adapter_catalog",
            AsyncMock(return_value=self._catalog([], ["low", "medium", "max"])),
        )
        request = self._request()
        request.query = {"backend": ACP_BACKEND_CODEX}

        levels = json.loads((await agents.api_effort_levels(request)).body.decode())

        assert levels == ["low", "medium", "max"]

    @pytest.mark.asyncio
    async def test_a_failed_probe_leaves_the_sentinel_rather_than_erroring(self, monkeypatch):
        """Not installed / not signed in must not 5xx a read-only picker."""
        from kiro_crew.acp.adapter_catalog import AdapterCatalog
        from kiro_crew.dashboard.handlers import agents

        monkeypatch.setattr(
            agents, "_probe_adapter_catalog", AsyncMock(return_value=AdapterCatalog())
        )
        request = self._request()
        request.query = {"backend": ACP_BACKEND_CLAUDE}

        response = await agents.api_models(request)

        assert response.status == 200
        assert [row["model_name"] for row in json.loads(response.body.decode())] == ["auto"]


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
