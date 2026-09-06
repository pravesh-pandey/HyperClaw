"""OpenCode keeps its model namespace, session store, and sandbox boundary."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.acp.client import AcpClient, AcpError, AcpModelUnavailable
from kiro_crew.acp.types import (
    ACP_BACKEND_CLAUDE,
    ACP_BACKEND_CODEX,
    ACP_BACKEND_OPENCODE,
    ACP_BACKENDS_INTERNAL_SANDBOX,
    ACP_BACKENDS_SESSION_SHARING,
    PROVIDER_LABEL_OPENCODE,
)
from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.providers.acp import AcpProvider, provider_label


def _model_options():
    return [
        {
            "id": "model",
            "currentValue": "local/small-model",
            "options": [
                {"value": "local/small-model", "name": "Local small model"},
                {"value": "host/large-model", "name": "Hosted planner"},
            ],
        },
        {"id": "effort", "options": [{"value": "low"}, {"value": "high"}]},
    ]


def test_opencode_models_and_identity_are_not_kiro(tmp_path):
    provider = AcpProvider(work_dir=tmp_path, acp_backend=ACP_BACKEND_OPENCODE)
    provider.client._store_session_config({"configOptions": _model_options()})

    assert provider_label(provider) == PROVIDER_LABEL_OPENCODE
    assert provider.client.current_model_id() == "local/small-model"
    assert [row["modelId"] for row in provider.available_models()] == [
        "local/small-model",
        "host/large-model",
    ]
    assert provider.get_valid_effort_levels() == ["low", "high"]
    assert ACP_BACKEND_OPENCODE not in ACP_BACKENDS_INTERNAL_SANDBOX
    assert ACP_BACKEND_OPENCODE not in ACP_BACKENDS_SESSION_SHARING


@pytest.mark.asyncio
async def test_opencode_model_switch_preserves_wire_id(tmp_path):
    client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_OPENCODE)
    client._session_id = "session"
    client._store_session_config({"configOptions": _model_options()})
    client._send_request = AsyncMock(return_value=1)
    client._wait_for_response = AsyncMock(return_value={"configOptions": _model_options()})

    await client.set_model("host/large-model")

    client._send_request.assert_awaited_once_with(
        "session/set_config_option",
        {"sessionId": "session", "configId": "model", "value": "host/large-model"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", [ACP_BACKEND_CODEX, ACP_BACKEND_OPENCODE])
async def test_unavailable_adapter_pin_never_uses_expensive_default(tmp_path, backend):
    client = AcpClient(work_dir=tmp_path, acp_backend=backend, model="local/removed-model")
    client._session_id = "session"
    client._store_session_config({"configOptions": _model_options()})
    client._send_request = AsyncMock()

    with pytest.raises(AcpModelUnavailable):
        await client._apply_startup_model()

    client._send_request.assert_not_awaited()
    assert client._model == "local/removed-model"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", [ACP_BACKEND_CODEX, ACP_BACKEND_OPENCODE])
async def test_adapter_resume_accepts_config_options_without_modes(tmp_path, backend):
    client = AcpClient(work_dir=tmp_path, acp_backend=backend)
    client.set_resume_session_id("saved-session")
    client._send_request = AsyncMock(side_effect=[1, 2])
    client._wait_for_response = AsyncMock(
        side_effect=[
            {"agentCapabilities": {"loadSession": True}},
            {"configOptions": _model_options()},
        ]
    )
    client._drain_notifications = AsyncMock()
    client._pooled_mcp_servers = lambda: []

    await client._initialize_session()

    assert client.resumed
    assert client._session_id == "saved-session"
    assert [call.args[0] for call in client._send_request.await_args_list] == [
        "initialize",
        "session/load",
    ]
    assert client._send_request.await_args_list[0].args[1]["protocolVersion"] == 1


def _request():
    return SimpleNamespace(
        app={
            "state": SimpleNamespace(
                _slots={}, sessions=SimpleNamespace(active_providers=lambda: [])
            )
        },
        query={"backend": ACP_BACKEND_OPENCODE},
        headers={},
    )


@pytest.mark.asyncio
async def test_cold_opencode_picker_discovers_models_without_kiro_login(monkeypatch):
    from kiro_crew.acp import opencode
    from kiro_crew.dashboard.handlers import agents

    cfg = KiroCrewConfig()
    monkeypatch.setattr(agents.KiroCrewConfig, "load", lambda: cfg)
    discover = AsyncMock(return_value=["local/small-model"])
    monkeypatch.setattr(opencode, "configured_model_ids", discover)
    login_gate = AsyncMock(side_effect=AssertionError("must not ask Kiro to authenticate"))
    monkeypatch.setattr(agents, "reject_if_kiro_unverified", login_gate)

    response = await agents.api_models(_request())

    assert response.status == 200
    assert [row["model_name"] for row in json.loads(response.body)] == ["auto", "local/small-model"]
    discover.assert_awaited_once()
    login_gate.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_model_discovery_is_retryable_not_cached_empty(monkeypatch):
    from kiro_crew.acp import opencode
    from kiro_crew.dashboard.handlers import agents

    monkeypatch.setattr(agents.KiroCrewConfig, "load", KiroCrewConfig)
    monkeypatch.setattr(
        opencode, "configured_model_ids", AsyncMock(side_effect=AcpError("missing"))
    )

    response = await agents.api_models(_request())

    assert response.status == 503
    assert json.loads(response.body)["code"] == "model_discovery_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [asyncio.TimeoutError, asyncio.CancelledError])
async def test_catalog_discovery_reaps_child_and_cleans_sandbox(tmp_path, monkeypatch, failure):
    from kiro_crew.acp import opencode

    cleanup = tmp_path / "sandbox-profile"
    cleanup.write_text("profile", encoding="utf-8")
    monkeypatch.setattr(opencode, "_resolve_opencode_bin", lambda: ("/bin/opencode", ""))
    prepare = AsyncMock(return_value=(["/bin/opencode", "models", "--pure"], {}, str(cleanup)))
    monkeypatch.setattr(opencode, "sandboxed_spawn_argv_async", prepare)
    monkeypatch.setattr(opencode, "finish_suspended_spawn", lambda *args, **kwargs: None)
    proc = MagicMock(returncode=None)
    proc.communicate = AsyncMock(side_effect=failure())
    spawn = AsyncMock(return_value=proc)
    monkeypatch.setattr(opencode, "create_subprocess_limited", spawn)
    reap = AsyncMock()
    monkeypatch.setattr(opencode.platform_compat, "kill_and_reap", reap)

    with pytest.raises(failure):
        await opencode.configured_model_ids(tmp_path, "standard")

    reap.assert_awaited_once_with(proc)
    assert not cleanup.exists()
    assert spawn.call_args.kwargs["cwd"] == str(tmp_path)
    assert prepare.call_args.kwargs["mode"] == "standard"


@pytest.mark.asyncio
async def test_a_failed_read_reports_the_childs_own_reason(tmp_path, monkeypatch):
    """A malformed opencode.jsonc must reach the operator, not a bare "failed".

    The child prints the offending file and key on stderr and exits non-zero.
    Discarding that leaves the one person who can fix it with nothing to act on.
    """
    from kiro_crew.acp import opencode

    monkeypatch.setattr(opencode, "_resolve_opencode_bin", lambda: ("/bin/opencode", ""))
    monkeypatch.setattr(
        opencode,
        "sandboxed_spawn_argv_async",
        AsyncMock(return_value=(["/bin/opencode", "models"], {}, None)),
    )
    monkeypatch.setattr(opencode, "finish_suspended_spawn", lambda *a, **k: None)
    proc = MagicMock(returncode=1)
    proc.communicate = AsyncMock(
        return_value=(b"", b"Error: Configuration is invalid at /cfg/opencode.jsonc\n")
    )
    monkeypatch.setattr(opencode, "create_subprocess_limited", AsyncMock(return_value=proc))

    with pytest.raises(AcpError) as excinfo:
        await opencode.configured_model_ids(tmp_path, "standard")

    assert "opencode.jsonc" in str(excinfo.value)


@pytest.mark.asyncio
async def test_the_catalog_read_matches_what_the_acp_session_serves(tmp_path, monkeypatch):
    """No ``--pure``: the session runs with plugins, so the catalog must too.

    A plugin-contributed provider is exactly the free/local model an operator
    picks this harness for; hiding it from the picker while the session serves
    it makes the two disagree.
    """
    from kiro_crew.acp import opencode

    monkeypatch.setattr(opencode, "_resolve_opencode_bin", lambda: ("/bin/opencode", ""))
    prepare = AsyncMock(return_value=(["/bin/opencode", "models"], {}, None))
    monkeypatch.setattr(opencode, "sandboxed_spawn_argv_async", prepare)
    monkeypatch.setattr(opencode, "finish_suspended_spawn", lambda *a, **k: None)
    proc = MagicMock(returncode=0)
    proc.communicate = AsyncMock(
        return_value=(b"opencode/free-model\nlocal/tiny\nopencode/free-model\n", b"")
    )
    monkeypatch.setattr(opencode, "create_subprocess_limited", AsyncMock(return_value=proc))

    ids = await opencode.configured_model_ids(tmp_path, "standard")

    assert "--pure" not in prepare.call_args.args[0]
    # Deduped, order preserved.
    assert ids == ["opencode/free-model", "local/tiny"]


@pytest.mark.asyncio
async def test_the_offline_read_never_runs_in_the_operators_home(monkeypatch):
    """Home is neither where a session runs nor whose config should be read."""
    from kiro_crew.acp import opencode
    from kiro_crew.dashboard.handlers import agents

    cfg = KiroCrewConfig()
    monkeypatch.setattr(agents.KiroCrewConfig, "load", lambda: cfg)
    seen: list = []

    async def _capture(work_dir, sandbox_mode):
        seen.append(work_dir)
        return ["local/tiny"]

    monkeypatch.setattr(opencode, "configured_model_ids", _capture)

    response = await agents.api_models(_request())

    assert response.status == 200
    assert seen and seen[0] != Path.home()


@pytest.mark.asyncio
async def test_the_catalog_read_does_not_wrap_the_cgroup_scope_twice(tmp_path, monkeypatch):
    """``sandboxed_spawn_argv`` already applies the scope; a second one is fatal.

    Wrapping again execs ``systemd-run --user --scope`` INSIDE the scope the
    first one just created, and systemd refuses the nested unit with "Unit
    run-p<pid>-i<n>.scope was already loaded or has a fragment file". Nothing
    about that is visible off a systemd host -- ``cgroup_scope_argv`` is a no-op
    everywhere else -- so it reached an operator as a permanent 503 on the
    OpenCode picker. Asserted on the argv actually spawned, because that is the
    only place the duplication is observable.
    """
    from kiro_crew.acp import opencode

    monkeypatch.setattr(opencode, "_resolve_opencode_bin", lambda: ("/bin/opencode", ""))
    # What the chokepoint really returns on a cgroup-capable host: already wrapped.
    already_wrapped = [
        "/usr/bin/systemd-run",
        "--user",
        "--scope",
        "-q",
        "--slice=kirocrew-agents.slice",
        "--",
        "/bin/opencode",
        "models",
    ]
    monkeypatch.setattr(
        opencode,
        "sandboxed_spawn_argv_async",
        AsyncMock(return_value=(already_wrapped, {}, None)),
    )
    monkeypatch.setattr(opencode, "finish_suspended_spawn", lambda *a, **k: None)
    spawned: list = []

    async def _spawn(*argv, **kwargs):
        spawned.append(list(argv))
        proc = MagicMock(returncode=0)
        proc.communicate = AsyncMock(return_value=(b"local/tiny\n", b""))
        return proc

    monkeypatch.setattr(opencode, "create_subprocess_limited", _spawn)

    await opencode.configured_model_ids(tmp_path, "standard")

    assert spawned, "the catalog read never spawned"
    assert spawned[0].count("/usr/bin/systemd-run") == 1
    assert spawned[0] == already_wrapped


@pytest.mark.asyncio
async def test_kiro_credential_never_reaches_the_foreign_catalog_child(tmp_path, monkeypatch):
    """KIRO_API_KEY authenticates a loop this child does not run.

    It is deliberately outside ``sandbox._AGENT_DENIED_ENV_KEYS``, so the raw
    environ snapshot would carry it into a foreign harness unless stripped here.
    """
    from kiro_crew.acp import opencode

    monkeypatch.setenv("KIRO_API_KEY", "sk-must-not-travel")
    monkeypatch.setattr(opencode, "_resolve_opencode_bin", lambda: ("/bin/opencode", ""))
    prepare = AsyncMock(return_value=(["/bin/opencode", "models"], {}, None))
    monkeypatch.setattr(opencode, "sandboxed_spawn_argv_async", prepare)
    monkeypatch.setattr(opencode, "finish_suspended_spawn", lambda *a, **k: None)
    proc = MagicMock(returncode=0)
    proc.communicate = AsyncMock(return_value=(b"local/tiny\n", b""))
    monkeypatch.setattr(opencode, "create_subprocess_limited", AsyncMock(return_value=proc))

    await opencode.configured_model_ids(tmp_path, "standard")

    # The env handed to the sandbox chokepoint is what the child inherits.
    assert "KIRO_API_KEY" not in prepare.call_args.kwargs["env"]


@pytest.mark.asyncio
# Kiro/KAS are excluded deliberately: their ``session/set_model`` is fired
# without awaiting a response, so a refusal never surfaces here. Making that
# path await one would add a step to the first-class harness (harness-parity).
@pytest.mark.parametrize("backend", [ACP_BACKEND_CLAUDE, ACP_BACKEND_CODEX, ACP_BACKEND_OPENCODE])
async def test_a_refused_startup_model_degrades_instead_of_killing_the_session(tmp_path, backend):
    """A harness that ANSWERS "I cannot serve that model" must not end the session.

    The startup model comes from the agent spec or a config default, not from a
    pick made for this turn. A Kiro agent spec pinning a GPT model while chat
    runs on Claude reaches the wire as an id Claude has no equivalent for, and
    failing there kills every turn with a stack trace over a setting the user
    never made for that session.
    """
    client = AcpClient(work_dir=tmp_path, acp_backend=backend, model="gpt-5.6-sol")
    client._session_id = "session"
    # No advertised set => entitlement unknown => the pre-flight allows it, so
    # the refusal can only come back from the harness itself.
    client._send_request = AsyncMock(return_value=1)
    client._wait_for_response = AsyncMock(
        side_effect=AcpError(
            "JSON-RPC error: {'code': -32603, 'data': "
            "{'details': 'Invalid value for config option model: gpt-5.6-sol'}}"
        )
    )

    await client._apply_startup_model()

    # Session survives, and the refused id is not left behind for the warm-pool
    # re-apply path to offer again.
    assert client._model == "auto"


@pytest.mark.asyncio
async def test_a_dead_process_during_startup_model_still_propagates(tmp_path):
    """The withhold must not swallow a broken session.

    A refusal means the harness answered; a death or a timeout means it did
    not. Reporting the second as "model withheld" would hand back a session
    that cannot serve a turn.
    """
    from kiro_crew.acp.client import AcpProcessDied

    client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_CLAUDE, model="gpt-5.6-sol")
    client._session_id = "session"
    client._send_request = AsyncMock(return_value=1)
    client._wait_for_response = AsyncMock(side_effect=AcpProcessDied("process exited"))

    with pytest.raises(AcpProcessDied):
        await client._apply_startup_model()


@pytest.mark.asyncio
async def test_the_claude_picker_reads_the_catalog_the_adapter_actually_sends(tmp_path):
    """The empty-picker symptom, at its source: claude-agent-acp sends no ``models``.

    Its ``session/new`` result carries ``configOptions`` only, so a picker fed
    from ``models.availableModels`` had nothing to show and stayed on "Auto"
    forever however healthy the session was. The payload here is the shape the
    installed adapter really returns.

    The ids reach the picker VERBATIM, because ``set_config_option`` accepts
    exactly these values -- folding ``opus`` to a registry key would offer a row
    the adapter then refuses.
    """
    from kiro_crew.dashboard.handlers.agents import _advertised_cc_models

    client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_CLAUDE, model="auto")
    client._session_id = "session"
    session_new = {
        "sessionId": "session",
        "configOptions": [
            {
                "id": "model",
                "name": "Model",
                "type": "select",
                "currentValue": "opus",
                "options": [
                    {"value": "default", "name": "Default (recommended)"},
                    {"value": "sonnet", "name": "Sonnet"},
                    {"value": "opus", "name": "Opus"},
                    {"value": "haiku", "name": "Haiku"},
                ],
            }
        ],
    }
    client._capture_available_models(session_new)
    client._store_session_config(session_new)

    provider = AcpProvider(work_dir=tmp_path, acp_backend=ACP_BACKEND_CLAUDE)
    provider._client = client
    request = SimpleNamespace(
        app={
            "state": SimpleNamespace(sessions=SimpleNamespace(active_providers=lambda: [provider]))
        }
    )
    names = [row["model_name"] for row in _advertised_cc_models(request)]

    assert names == ["default", "sonnet", "opus", "haiku"]
    # The session also knows which of them it is running, which the missing
    # ``models`` block never told it.
    assert client._resolved_model_id == "opus"
