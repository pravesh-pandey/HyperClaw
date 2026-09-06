"""Read OpenCode's configured model catalog without starting a model turn.

The picker needs an OpenCode model list before any session exists: an operator
choosing OpenCode for the ``background`` role is configuring a harness they have
not opened a chat on, so there is no live session to read an advertised set
from. ``opencode models`` answers that offline, printing one ``provider/model``
wire id per line — exactly the ids ``session/set_config_option`` accepts, so the
picker and the wire cannot disagree.

Plugins are deliberately NOT disabled for this read. The session Crew spawns is
``opencode acp`` with plugins active, so a ``--pure`` catalog would omit any
provider a plugin contributes — silently hiding the local or free model the
operator configured that way, which is the whole reason they picked this
harness.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from kiro_crew import platform_compat
from kiro_crew.acp.client import AcpError, _resolve_opencode_bin, finish_suspended_spawn
from kiro_crew.config.loader import strip_kiro_cli_api_key
from kiro_crew.env import augmented_path
from kiro_crew.sandbox import (
    create_subprocess_limited,
    sandboxed_spawn_argv_async,
)
from kiro_crew.security import redact_credentials, redact_exfiltration_urls

#: Bound on a cold catalog read. Generous because the first spawn of a
#: standalone binary on a cold page cache is the slow case, and the caller
#: answers 503 (client retries with backoff) rather than blocking a turn.
MODEL_LIST_TIMEOUT_SECONDS = 20.0

#: How much of a failing read's stderr reaches the log. Bounded because the
#: child's output is foreign text, and redacted for the same reason.
_STDERR_TAIL_CHARS = 1000


def _diagnostic(raw: bytes) -> str:
    """Bounded, redacted tail of a foreign child's stderr."""
    text, _ = redact_exfiltration_urls(raw.decode("utf-8", errors="replace").strip())
    text, _ = redact_credentials(text)
    return text[-_STDERR_TAIL_CHARS:]


async def configured_model_ids(work_dir: Path, sandbox_mode: str) -> list[str]:
    """List OpenCode's wire model ids, under the chat path's sandbox posture.

    No prompt is sent and no login is triggered: this reads configuration the
    operator already wrote. The posture is the CONFIGURED one rather than a
    pinned tier, for the same reason the kiro one-shot reads use it — a catalog
    read of a binary must not run under a different confinement than the chat
    session that spawns the same binary.

    Raises ``AcpError`` with the child's own diagnostic when the read fails, so
    a malformed ``opencode.jsonc`` reaches the operator as the reason rather
    than as a bare "discovery failed".
    """
    executable, search_path = await asyncio.to_thread(_resolve_opencode_bin)
    if executable is None:
        raise AcpError(f"opencode not found (searched {search_path})")

    env = dict(os.environ)
    env["PATH"] = augmented_path(env.get("PATH", ""))
    # kiro-cli's own model credential authenticates a loop this child does not
    # run, and it is deliberately outside sandbox._AGENT_DENIED_ENV_KEYS, so the
    # raw environ snapshot would otherwise carry it into a foreign harness.
    strip_kiro_cli_api_key(env)

    argv, env, cleanup = await sandboxed_spawn_argv_async(
        [executable, "models"], mode=sandbox_mode, env=env, strip_python_env=True
    )
    # NOT re-wrapped with ``cgroup_scope_argv``: ``sandboxed_spawn_argv`` already
    # applies it as its OUTERMOST layer, so a second wrap execs
    # ``systemd-run --scope`` inside the scope the first one just created. systemd
    # refuses that with "Unit run-p<pid>-i<n>.scope was already loaded or has a
    # fragment file", which surfaced as a permanent 503 on the OpenCode picker.
    # The sites that DO call it pair it with bare ``wrap_argv``, which does not.
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await create_subprocess_limited(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(work_dir),
            env=env,
            start_new_session=platform_compat.IS_POSIX,
            creationflags=(
                platform_compat.CREATE_NEW_PROCESS_GROUP | platform_compat.CREATE_SUSPENDED
            ),
        )
        await asyncio.to_thread(finish_suspended_spawn, proc, proc.pid, label="opencode models")
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=MODEL_LIST_TIMEOUT_SECONDS
        )
        if proc.returncode:
            raise AcpError(
                f"opencode models exited {proc.returncode}: "
                f"{_diagnostic(stderr) or '<no stderr>'}"
            )
        # One bare ``provider/model`` id per line. A line carrying whitespace is
        # prose (a banner, a warning), never an id, so it is the discriminator
        # rather than a position or a count.
        ids = list(
            dict.fromkeys(
                line.strip()
                for line in stdout.decode("utf-8", errors="replace").splitlines()
                if "/" in line and not any(char.isspace() for char in line.strip())
            )
        )
        if not ids:
            raise AcpError("opencode models reported no configured models")
        return ids
    finally:
        if proc is not None and proc.returncode is None:
            await platform_compat.kill_and_reap(proc)
        if cleanup:
            await asyncio.to_thread(Path(cleanup).unlink, missing_ok=True)
