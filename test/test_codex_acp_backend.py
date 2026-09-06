"""Codex ACP adapter selection, model configuration, and effort semantics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro_crew.acp.client import (
    _CODEX_ACP_PKG_ENTRY,
    PROTOCOL_VERSION_ADAPTER,
    AcpClient,
    _resolve_vendored_codex_acp,
)
from kiro_crew.acp.types import ACP_BACKEND_CODEX
from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.providers.acp import AcpProvider


def test_project_local_codex_adapter_is_resolved(tmp_path: Path) -> None:
    entry = tmp_path / "_vendor" / "node_modules" / _CODEX_ACP_PKG_ENTRY
    entry.parent.mkdir(parents=True)
    entry.write_text("// adapter\n", encoding="utf-8")

    assert _resolve_vendored_codex_acp(tmp_path) == str(entry)


@pytest.mark.asyncio
async def test_codex_uses_numeric_acp_protocol(tmp_path: Path) -> None:
    client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_CODEX)
    client._session_id = "session-1"
    sent: list[tuple[str, dict]] = []

    async def send(method: str, params: dict) -> int:
        sent.append((method, params))
        return 1

    client._send_request = send  # type: ignore[assignment]
    client._wait_for_response = AsyncMock(  # type: ignore[method-assign]
        return_value={"protocolVersion": PROTOCOL_VERSION_ADAPTER, "agentCapabilities": {}}
    )
    client._drain_notifications = AsyncMock()  # type: ignore[method-assign]

    await client._initialize_session()

    assert sent[0][0] == "initialize"
    assert sent[0][1]["protocolVersion"] == PROTOCOL_VERSION_ADAPTER
    assert all(method != "session/set_mode" for method, _params in sent)


@pytest.mark.asyncio
async def test_codex_model_switch_uses_session_config(tmp_path: Path) -> None:
    client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_CODEX)
    client._session_id = "session-1"
    client.set_config_option = AsyncMock()  # type: ignore[method-assign]

    await client.set_model("model-from-adapter")

    client.set_config_option.assert_awaited_once_with("model", "model-from-adapter")


@pytest.mark.asyncio
async def test_codex_effort_alias_uses_advertised_wire_id(tmp_path: Path) -> None:
    client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_CODEX)
    client._session_id = "session-1"
    options = [
        {
            "id": "reasoning_effort",
            "category": "thought_level",
            "options": [{"value": "low"}, {"value": "high"}],
        }
    ]
    client._acp_config_options = options
    client._send_request = AsyncMock(return_value=7)  # type: ignore[method-assign]
    client._wait_for_response = AsyncMock(  # type: ignore[method-assign]
        return_value={"configOptions": options}
    )

    assert client.supports_config_option("effort") is True
    assert client.get_valid_effort_levels() == ["low", "high"]

    await client.set_config_option("effort", "high")

    client._send_request.assert_awaited_once_with(
        "session/set_config_option",
        {"sessionId": "session-1", "configId": "reasoning_effort", "value": "high"},
    )


def test_codex_auto_defers_global_effort_until_model_is_advertised(tmp_path: Path) -> None:
    config = KiroCrewConfig()
    config.agent.acp_backend = ACP_BACKEND_CODEX
    config.agent.reasoning_effort = "high"

    with patch("kiro_crew.providers.acp.AcpProvider") as mock_provider:
        mock_provider.return_value = MagicMock()
        config.create_provider_factory()(cwd=str(tmp_path), session_key="dashboard:1")

    assert mock_provider.call_args.kwargs["effort_per_model"] == {}
    assert mock_provider.call_args.kwargs["default_effort"] == "high"


def test_codex_deferred_effort_resolves_from_live_model_option(tmp_path: Path) -> None:
    provider = AcpProvider(
        work_dir=tmp_path,
        acp_backend=ACP_BACKEND_CODEX,
        default_effort="high",
    )
    provider.client._acp_config_options = [
        {"id": "model", "currentValue": "gpt-5.6-sol"},
        {
            "id": "reasoning_effort",
            "options": [{"value": "low"}, {"value": "high"}],
        },
    ]

    assert provider._resolve_effort() == "high"
