"""Crew MCP exposure for external ACP adapters."""

from __future__ import annotations

import pytest

import kiro_crew.acp.client as acp_client
from kiro_crew.acp.client import AcpClient
from kiro_crew.acp.types import ACP_BACKEND_CLAUDE, ACP_BACKEND_CODEX


def _managed_servers(monkeypatch):
    monkeypatch.setattr(
        acp_client,
        "_MANAGED_MCP_SERVERS",
        {
            "kirocrew-core": {
                "invocation_fn": lambda: ("/usr/bin/kirocrew", ["mcp-core"]),
            },
            "kirocrew-cron": {
                "invocation_fn": lambda: ("/usr/bin/kirocrew", ["mcp-cron"]),
            },
            "kirocrew-dashboard": {
                "invocation_fn": lambda: ("/usr/bin/kirocrew", ["mcp-dashboard"]),
                "opt_in": True,
            },
        },
    )


def test_external_adapters_receive_only_always_emitted_crew_servers(tmp_path, monkeypatch):
    _managed_servers(monkeypatch)

    claude = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_CLAUDE)
    codex = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_CODEX)

    assert [entry["name"] for entry in claude._claude_session_mcp_servers()] == [
        "kirocrew-core",
        "kirocrew-cron",
    ]
    assert codex._adapter_session_mcp_servers() == claude._claude_session_mcp_servers()
    assert all("autoApprove" not in entry for entry in codex._adapter_session_mcp_servers())


def test_claude_session_options_disable_native_subagents_and_project_settings(tmp_path):
    client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_CLAUDE)

    options = client._claude_session_options()

    assert options["settingSources"] == []
    assert options["disallowedTools"] == ["Agent", "Task"]
    assert options["settings"]["permissions"]["ask"] == [
        "mcp__kirocrew-core__*",
        "mcp__kirocrew-cron__*",
    ]


@pytest.mark.asyncio
async def test_claude_session_new_carries_crew_mcp_and_security_options(tmp_path, monkeypatch):
    _managed_servers(monkeypatch)
    client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_CLAUDE)
    sent: list[tuple[str, dict]] = []

    async def send(method, params):
        sent.append((method, params))
        return len(sent)

    async def wait(_request_id, timeout=0.0, *, method="", expected_mcp=None):
        return {"sessionId": "session-1"}

    client._send_request = send  # type: ignore[assignment]
    client._wait_for_response = wait  # type: ignore[assignment]

    response = await client._new_session_following_substitution()

    assert response["sessionId"] == "session-1"
    params = sent[0][1]
    assert [entry["name"] for entry in params["mcpServers"]] == [
        "kirocrew-core",
        "kirocrew-cron",
    ]
    assert params["_meta"]["claudeCode"]["options"] == client._claude_session_options()
