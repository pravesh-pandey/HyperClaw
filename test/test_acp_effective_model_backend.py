"""The cold-start model id lands in the namespace of the backend that runs it.

``acp_effective_model`` is the factory's own selection. It used to hardcode
``to_acp_id``, so a concrete ``agent.model`` reached an external adapter in
kiro's namespace, which ``set_config_option`` rejects. The warm-pool switch path
(``session_allocation``) keyed on the backend, so the same pinned model behaved
differently depending on whether a pooled process happened to exist. These pin
that the two paths agree.

For a member of ``ACP_BACKENDS_OWN_MODEL_CATALOG`` "its namespace" means the id
UNTRANSLATED: those adapters advertise the values they accept, so a registry
translation is what lands outside the accepted set —
``to_provider_id("opus", "claude_code")`` yields
``global.anthropic.claude-opus-4-8[1m]``, which claude-agent-acp refuses with a
``-32603``.
"""

from __future__ import annotations

from kiro_crew.acp_backends import ACP_BACKEND_CLAUDE, ACP_BACKEND_KAS, ACP_BACKEND_KIRO
from kiro_crew.config.loader import AgentConfig, KiroCrewConfig

# A canonical registry key whose kiro id and claude provider id differ — the
# whole point of the fix. Kept as data so the assertions below read as "which
# namespace did it land in", not as a second copy of the registry.
PINNED = "opus-4.8-1m"
KIRO_ID = "claude-opus-4.8"


def _cfg(backend: str, model: str) -> KiroCrewConfig:
    """A config resolved without reading the installed agents dir.

    ``model`` is written straight onto ``AgentConfig`` so the global is already
    collapsed — ``_resolve_agent_model`` (which reads
    ``~/.kiro/agents/kirocrew.json``) is only consulted for the ``auto``
    sentinel, and the auto case below wants exactly that skipped.
    """
    return KiroCrewConfig(agent=AgentConfig(acp_backend=backend, model=model))


def test_claude_backend_keeps_the_id_untranslated() -> None:
    """The configured id reaches the adapter as written — no registry rewrite."""
    resolved = _cfg(ACP_BACKEND_CLAUDE, PINNED).acp_effective_model(None, None)
    assert resolved != KIRO_ID, "kiro-namespaced id leaked to the claude backend"
    assert resolved == PINNED, f"expected the id verbatim, got {resolved!r}"
    assert "anthropic" not in resolved, "a provider-id translation is what the adapter refuses"


def test_kiro_backend_keeps_the_kiro_id() -> None:
    """The kiro path is unchanged — the fix must not retranslate it."""
    assert _cfg(ACP_BACKEND_KIRO, PINNED).acp_effective_model(None, None) == KIRO_ID


def test_kas_backend_keeps_the_kiro_id() -> None:
    """kas is served through the same ids as kiro, so it takes the same branch."""
    assert _cfg(ACP_BACKEND_KAS, PINNED).acp_effective_model(None, None) == KIRO_ID


def test_auto_pins_nothing_on_the_claude_backend() -> None:
    """The default config is the path that already worked — keep it sending nothing.

    ``auto`` collapses to ``""`` in both namespaces and the client skips the
    model send for ``""``. This is why enabling claude via config.json works
    today, and why the defect above was never the mainline path.
    """
    assert _cfg(ACP_BACKEND_CLAUDE, "").acp_effective_model(None, None) == ""


def test_an_explicit_override_also_reaches_claude_untranslated() -> None:
    """``model_override`` wins the precedence chain and takes the same branch."""
    resolved = _cfg(ACP_BACKEND_CLAUDE, "").acp_effective_model(None, PINNED)
    assert resolved != KIRO_ID
    assert resolved == PINNED


def test_a_kiro_agent_spec_model_never_collapses_onto_claude() -> None:
    """``auto`` must NOT be expanded from ``~/.kiro/agents/kirocrew.json`` here.

    That file is kiro-cli's, so the model in it is whatever the operator runs
    kiro-cli on. Expanding it for an own-catalog harness is how ``gpt-5.6-sol``
    reached claude-agent-acp and was refused with a ``-32603``.
    """
    from unittest.mock import patch

    from kiro_crew.config import loader as loader_mod

    with patch.object(loader_mod.KiroCrewConfig, "_resolve_agent_model") as resolve:
        resolve.return_value = "gpt-5.6-sol"
        assert _cfg(ACP_BACKEND_CLAUDE, "auto").acp_effective_model(None, None) == ""
        resolve.assert_not_called()
        # kiro-cli still reads its own file.
        assert _cfg(ACP_BACKEND_KIRO, "auto").acp_effective_model(None, None) == "gpt-5.6-sol"
