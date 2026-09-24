"""Per-task-class harness selection (agent.role_backends).

The point of the feature is to run unattended work on a cheaper harness than
interactive chat — Claude for the chat you watch, Codex for background and
sub-agents. These pin the coercion, the inherit rule that makes an unset role
follow chat, and the two spawn paths that have to honour a pin for it to mean
anything.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.acp_backends import ACP_BACKEND_CLAUDE, ACP_BACKEND_CODEX, ACP_BACKEND_KIRO
from kiro_crew.config.loader import AgentConfig, KiroCrewConfig
from kiro_crew.config.sections import ROLE_BACKEND_KEYS, coerce_role_backends


class TestCoerceRoleBackends:
    def test_keeps_known_roles_that_name_a_servable_harness(self) -> None:
        out = coerce_role_backends(
            {"background": ACP_BACKEND_CODEX, "subagent": ACP_BACKEND_CLAUDE, "bogus": "x"}
        )
        assert out == {"background": ACP_BACKEND_CODEX, "subagent": ACP_BACKEND_CLAUDE}

    def test_explicit_empty_is_retained_as_a_kiro_pin(self) -> None:
        # An absent role inherits chat; an explicitly empty role selects the
        # Kiro harness, whose wire id is the empty string.
        assert coerce_role_backends({"background": "", "subagent": "   "}) == {
            "background": "",
            "subagent": "",
        }

    def test_unknown_backend_is_dropped_instead_of_becoming_kiro(self) -> None:
        assert coerce_role_backends({"background": "harness-from-a-later-build"}) == {}

    @pytest.mark.parametrize("raw", [None, "nope", 7, []])
    def test_non_dict_is_empty(self, raw: object) -> None:
        assert coerce_role_backends(raw) == {}

    def test_post_init_normalizes_a_hand_built_config(self) -> None:
        a = AgentConfig(role_backends={"background": ACP_BACKEND_CODEX, "x": "y"})
        assert a.role_backends == {"background": ACP_BACKEND_CODEX}

    def test_roles_match_the_model_pin_roles(self) -> None:
        # The Settings rows pair a model and a harness per role; a role that
        # could take one but not the other would render a half-configurable row.
        from kiro_crew.config.sections import ROLE_MODEL_KEYS

        assert ROLE_BACKEND_KEYS == ROLE_MODEL_KEYS


class TestResolveBackend:
    def test_an_unset_role_inherits_the_chat_harness(self) -> None:
        # Deliberately UNLIKE resolve_model, which does not inherit agent.model:
        # a model has an "auto" the provider can resolve, a harness does not.
        a = AgentConfig(acp_backend=ACP_BACKEND_CLAUDE)
        assert a.resolve_backend("background") == ACP_BACKEND_CLAUDE
        assert a.resolve_backend("subagent") == ACP_BACKEND_CLAUDE

    def test_a_pinned_role_overrides_chat(self) -> None:
        a = AgentConfig(
            acp_backend=ACP_BACKEND_CLAUDE,
            role_backends={"background": ACP_BACKEND_CODEX},
        )
        assert a.resolve_backend("background") == ACP_BACKEND_CODEX
        assert a.resolve_backend("subagent") == ACP_BACKEND_CLAUDE

    def test_an_explicit_kiro_role_overrides_non_kiro_chat(self) -> None:
        a = AgentConfig(acp_backend=ACP_BACKEND_CLAUDE, role_backends={"background": ""})
        assert a.resolve_backend("background") == ACP_BACKEND_KIRO
        assert a.resolve_backend("subagent") == ACP_BACKEND_CLAUDE

    def test_the_headline_configuration_resolves_as_advertised(self) -> None:
        # Claude for chat, Codex for the cheap unattended work.
        a = AgentConfig(
            acp_backend=ACP_BACKEND_CLAUDE,
            role_backends={"background": ACP_BACKEND_CODEX, "subagent": ACP_BACKEND_CODEX},
        )
        assert a.acp_backend == ACP_BACKEND_CLAUDE
        assert a.resolve_backend("background") == ACP_BACKEND_CODEX
        assert a.resolve_backend("subagent") == ACP_BACKEND_CODEX

    def test_an_unknown_role_inherits_rather_than_raising(self) -> None:
        a = AgentConfig(acp_backend=ACP_BACKEND_CODEX)
        assert a.resolve_backend("nope") == ACP_BACKEND_CODEX


def test_config_round_trip_preserves_role_backends(tmp_path, monkeypatch) -> None:
    import json

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "agent": {
                    "acp_backend": ACP_BACKEND_CLAUDE,
                    "role_backends": {"background": "", "subagent": ACP_BACKEND_CODEX},
                }
            }
        )
    )
    monkeypatch.setattr("kiro_crew.config.loader.config_path", lambda: cfg_file)
    cfg = KiroCrewConfig.load()
    assert cfg.agent.role_backends == {"background": "", "subagent": ACP_BACKEND_CODEX}
    assert cfg.to_dict()["agent"]["role_backends"] == {
        "background": "",
        "subagent": ACP_BACKEND_CODEX,
    }


class TestSubagentPin:
    """``None`` means "omit the kwarg"; a concrete value must reach the factory."""

    @staticmethod
    def _cfg(monkeypatch, chat: str, pinned: dict) -> None:
        monkeypatch.setattr(
            "kiro_crew.config.loader.KiroCrewConfig.load",
            classmethod(
                lambda cls: KiroCrewConfig(
                    agent=AgentConfig(acp_backend=chat, role_backends=pinned)
                )
            ),
        )

    def test_unpinned_role_answers_none(self, monkeypatch) -> None:
        from kiro_crew import subagent

        self._cfg(monkeypatch, ACP_BACKEND_CLAUDE, {})
        assert subagent._subagent_default_backend() is None

    def test_a_pin_matching_chat_answers_none(self, monkeypatch) -> None:
        # Nothing to override, so the kwarg stays absent and the factory path is
        # byte-identical to a build without this feature.
        from kiro_crew import subagent

        self._cfg(monkeypatch, ACP_BACKEND_CODEX, {"subagent": ACP_BACKEND_CODEX})
        assert subagent._subagent_default_backend() is None

    def test_a_differing_pin_is_returned(self, monkeypatch) -> None:
        from kiro_crew import subagent

        self._cfg(monkeypatch, ACP_BACKEND_CLAUDE, {"subagent": ACP_BACKEND_CODEX})
        assert subagent._subagent_default_backend() == ACP_BACKEND_CODEX

    def test_an_explicit_kiro_pin_is_returned_when_chat_is_external(self, monkeypatch) -> None:
        from kiro_crew import subagent

        self._cfg(monkeypatch, ACP_BACKEND_CLAUDE, {"subagent": ACP_BACKEND_KIRO})
        assert subagent._subagent_default_backend() == ACP_BACKEND_KIRO

    def test_unpinned_role_inherits_a_per_slot_parent_backend(self, monkeypatch) -> None:
        from kiro_crew import subagent

        self._cfg(monkeypatch, ACP_BACKEND_KIRO, {})
        assert subagent._subagent_default_backend(ACP_BACKEND_CLAUDE) == ACP_BACKEND_CLAUDE

    def test_an_unreadable_config_answers_none_rather_than_raising(self, monkeypatch) -> None:
        from kiro_crew import subagent

        monkeypatch.setattr(
            "kiro_crew.config.loader.KiroCrewConfig.load",
            classmethod(lambda cls: (_ for _ in ()).throw(OSError("unreadable"))),
        )
        assert subagent._subagent_default_backend() is None


class TestBackgroundPin:
    """The background runtime must spawn under the BACKGROUND role's harness."""

    @staticmethod
    def _runtime(chat: str, pinned: dict):
        from kiro_crew.session_background import BackgroundSessionRuntime

        owner = MagicMock()
        owner._cfg.agent = AgentConfig(acp_backend=chat, role_backends=pinned)
        deps = MagicMock()
        deps.acp_backend_kiro = ACP_BACKEND_KIRO
        runtime = BackgroundSessionRuntime.__new__(BackgroundSessionRuntime)
        runtime._owner = owner
        runtime._deps = deps
        return runtime

    def test_pinned_background_harness_is_used(self) -> None:
        runtime = self._runtime(ACP_BACKEND_CLAUDE, {"background": ACP_BACKEND_CODEX})
        assert runtime._configured_bg_backend_raw() == ACP_BACKEND_CODEX

    def test_unpinned_background_follows_chat(self) -> None:
        runtime = self._runtime(ACP_BACKEND_CLAUDE, {})
        assert runtime._configured_bg_backend_raw() == ACP_BACKEND_CLAUDE

    def test_a_subagent_pin_does_not_leak_into_background(self) -> None:
        # The two roles are independent; pinning one must not move the other.
        runtime = self._runtime(ACP_BACKEND_CLAUDE, {"subagent": ACP_BACKEND_CODEX})
        assert runtime._configured_bg_backend_raw() == ACP_BACKEND_CLAUDE

    @pytest.mark.asyncio
    async def test_provider_backed_background_session_receives_role_backend(self) -> None:
        from asyncio import Lock, Semaphore

        from kiro_crew.session_background import BackgroundRuntimeDeps, BackgroundSessionRuntime

        provider = MagicMock()
        provider.start = AsyncMock()
        owner = MagicMock()
        owner._cfg.agent = AgentConfig(
            acp_backend=ACP_BACKEND_CLAUDE,
            role_backends={"background": ACP_BACKEND_CODEX},
        )
        owner._provider_factory = MagicMock(return_value=provider)
        owner._lock = Lock()
        owner._start_sem = Semaphore(1)
        owner._closing = False
        owner._sessions = {}
        owner._configured_bg_backend = lambda: ACP_BACKEND_CODEX
        deps = MagicMock(spec=BackgroundRuntimeDeps)
        deps.background_key = "_bg"
        deps.background_agent = "kirocrew-lite"
        deps.acp_backend_kiro = ACP_BACKEND_KIRO
        deps.logger = MagicMock()
        deps.first_turn_nothing_armed = object()
        deps.session_factory = MagicMock(return_value=MagicMock())
        runtime = BackgroundSessionRuntime(owner, deps)

        await runtime._ensure_background()

        owner._provider_factory.assert_called_once_with(
            "_bg",
            agent="kirocrew-lite",
            acp_backend_override=ACP_BACKEND_CODEX,
        )

    @pytest.mark.asyncio
    async def test_cached_provider_background_session_is_replaced_after_role_switch(self) -> None:
        from asyncio import Lock, Semaphore

        from kiro_crew.session_background import BackgroundRuntimeDeps, BackgroundSessionRuntime

        old_provider = MagicMock()
        old_provider.client.backend = ACP_BACKEND_CLAUDE
        old_provider.shutdown = AsyncMock()
        old_session = MagicMock()
        old_session.provider = old_provider
        old_session.semaphore = Semaphore(1)
        new_provider = MagicMock()
        new_provider.start = AsyncMock()
        owner = MagicMock()
        owner._cfg.agent = AgentConfig(
            acp_backend=ACP_BACKEND_CLAUDE,
            role_backends={"background": ACP_BACKEND_CODEX},
        )
        owner._provider_factory = MagicMock(return_value=new_provider)
        owner._configured_bg_backend = lambda: ACP_BACKEND_CODEX
        owner._lock = Lock()
        owner._start_sem = Semaphore(1)
        owner._closing = False
        owner._sessions = {"_bg": old_session}
        deps = MagicMock(spec=BackgroundRuntimeDeps)
        deps.background_key = "_bg"
        deps.background_agent = "kirocrew-lite"
        deps.acp_backend_kiro = ACP_BACKEND_KIRO
        deps.logger = MagicMock()
        deps.first_turn_nothing_armed = object()
        deps.session_factory = MagicMock(return_value=MagicMock())
        runtime = BackgroundSessionRuntime(owner, deps)

        await runtime._ensure_background()

        old_provider.shutdown.assert_awaited_once()
        owner._provider_factory.assert_called_once_with(
            "_bg",
            agent="kirocrew-lite",
            acp_backend_override=ACP_BACKEND_CODEX,
        )


class TestWriteSchema:
    def test_role_pins_offer_exactly_the_chat_pin_values(self) -> None:
        # One derivation for both, so a harness an edition registers reaches the
        # role pins the moment it reaches the chat pin.
        from kiro_crew.dashboard.handlers.core import (
            _selectable_acp_backends,
            _selectable_role_backends,
        )

        assert _selectable_role_backends() == _selectable_acp_backends()

    @pytest.mark.parametrize("role", ROLE_BACKEND_KEYS)
    def test_each_role_has_a_writable_config_path(self, role: str) -> None:
        from kiro_crew.dashboard.handlers.core import _EDITABLE_CONFIG

        assert f"agent.role_backends.{role}" in _EDITABLE_CONFIG


class TestRoleModelValidationFollowsTheRoleHarness:
    """A role's model pin is validated against the ROLE's harness, not chat's.

    Without this a cross-harness pin — the whole point of role_backends — is
    rejected for being absent from the CHAT harness's advertised set.
    """

    @staticmethod
    def _request(backend: str, models: list[str]) -> MagicMock:
        provider = MagicMock()
        provider.client = MagicMock()
        provider.client.backend = backend
        provider.available_models = lambda: [{"modelId": m} for m in models]
        request = MagicMock()
        sessions = MagicMock()
        sessions.active_providers = lambda: [provider]
        request.app = {"state": MagicMock(sessions=sessions)}
        return request

    def test_the_role_harness_is_resolved_from_the_path(self, monkeypatch) -> None:
        from kiro_crew.dashboard.handlers import core

        cfg = KiroCrewConfig(
            agent=AgentConfig(
                acp_backend=ACP_BACKEND_KIRO,
                role_backends={"subagent": ACP_BACKEND_CODEX},
            )
        )
        monkeypatch.setattr(KiroCrewConfig, "load", classmethod(lambda cls: cfg))
        assert core._validation_backend_for("agent.role_models.subagent") == ACP_BACKEND_CODEX
        # An unpinned role still resolves — to the chat harness, via inherit.
        assert core._validation_backend_for("agent.role_models.background") == ACP_BACKEND_KIRO

    def test_a_non_role_path_expresses_no_opinion(self, monkeypatch) -> None:
        from kiro_crew.dashboard.handlers import core

        # agent.fallback_model shares the validator but is not a role pin, so it
        # must keep resolving its harness the way it always did.
        assert core._validation_backend_for("agent.fallback_model") is None

    def test_a_codex_pin_is_not_measured_against_kiros_catalog(self, monkeypatch) -> None:
        from kiro_crew.dashboard.handlers import core

        cfg = KiroCrewConfig(
            agent=AgentConfig(
                acp_backend=ACP_BACKEND_KIRO,
                role_backends={"subagent": ACP_BACKEND_CODEX},
            )
        )
        monkeypatch.setattr(KiroCrewConfig, "load", classmethod(lambda cls: cfg))
        # Only a KIRO session is live, advertising kiro ids. Validating the
        # Codex pin against that set is the rejection this guards against.
        request = self._request(ACP_BACKEND_KIRO, ["claude-opus-4.5"])
        backend = core._validation_backend_for("agent.role_models.subagent")
        assert core._validate_role_model("gpt-5.6-sol", request, backend=backend) is None

    def test_a_cross_harness_pin_on_an_INHERITING_role_names_the_harness(self, monkeypatch) -> None:
        """ "Not available on your account" is the wrong diagnosis, and it sticks.

        A role with no Provider of its own borrows the chat harness, so its pin
        is measured against THAT harness's catalog. Reporting the refusal as an
        entitlement problem sent an operator to their billing page when the fix
        was one control away — and because the write is refused, the role
        silently keeps running the chat harness's default, which is exactly how
        a subagent pinned to a GPT model came back reporting Claude.
        """
        from kiro_crew.dashboard.handlers import core

        cfg = KiroCrewConfig(agent=AgentConfig(acp_backend=ACP_BACKEND_CLAUDE, role_backends={}))
        monkeypatch.setattr(KiroCrewConfig, "load", classmethod(lambda cls: cfg))
        request = self._request(ACP_BACKEND_CLAUDE, ["sonnet", "opus"])

        reason = core._validate_role_model(
            "gpt-5.6-luna",
            request,
            backend=ACP_BACKEND_CLAUDE,
            path_key="agent.role_models.subagent",
        )

        assert reason is not None
        # Names the harness, the role, and the control that fixes it.
        assert "claude" in reason and "subagent" in reason
        assert "agent.role_backends.subagent" in reason
        assert "not available on your account" not in reason

    def test_a_pin_that_is_not_a_role_keeps_the_plain_entitlement_message(
        self, monkeypatch
    ) -> None:
        """The chat default has no role to redirect to, so nothing is invented."""
        from kiro_crew.dashboard.handlers import core

        cfg = KiroCrewConfig(agent=AgentConfig(acp_backend=ACP_BACKEND_CLAUDE))
        monkeypatch.setattr(KiroCrewConfig, "load", classmethod(lambda cls: cfg))
        request = self._request(ACP_BACKEND_CLAUDE, ["sonnet", "opus"])

        reason = core._validate_role_model(
            "gpt-5.6-luna", request, backend=ACP_BACKEND_CLAUDE, path_key="agent.model"
        )

        assert reason is not None and "not available on your account" in reason
        assert "role_backends" not in reason
