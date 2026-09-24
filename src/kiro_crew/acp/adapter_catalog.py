"""Read an external adapter's model + effort catalog with no live chat session.

The picker needs this BEFORE any session exists. That is not an edge case, it is
the setup case: an operator opening Settings to choose a harness, its default
model and its effort level has, by definition, never opened a chat on it.
kiro-cli answers offline through ``kiro-cli chat --list-models``; an adapted
harness whose model ids come only from what it advertises has no such command
and reports the account's real models and the model's effort ladder in
``session/new``'s ``configOptions``. Without this read, its picker could offer
only a static registry snapshot or the ``auto`` sentinel until a session
happened to run -- a snapshot that misses every model newer than it. So the
probe is a full ACP handshake — spawn, ``initialize``, ``session/new``, read,
tear down — driven through :class:`AcpClient` rather than hand-rolled JSON-RPC,
so it inherits the spawn path, sandbox posture, env scrubbing and process
teardown the chat session already uses and cannot drift from them.

No prompt is sent, so no tokens are spent and no model turn runs; this reads
configuration and entitlement the account already has.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from kiro_crew.acp_backends import (
    ACP_BACKENDS_ACP_RUNTIME,
    ACP_BACKENDS_ADVERTISED_MODEL_SELECTION,
)

logger = logging.getLogger(__name__)

#: Bound on one handshake. Generous because the slow case is a cold spawn of a
#: Node adapter on an unwarmed page cache; the caller degrades to the ``auto``
#: sentinel rather than blocking a picker, so overshooting costs a poll, not a
#: turn.
PROBE_TIMEOUT_SECONDS = 45.0

#: The harnesses this probe reads: those whose model ids come only from what they
#: advertise, AND that :class:`AcpClient` drives. A harness on the multiplexed
#: runtime (``ACP_BACKENDS_ACP_RUNTIME``, e.g. Codex) speaks its handshake dialect
#: through ``acp.harness``, never through this client, so a probe here would spawn
#: it only to be refused; its picker reads the advertised-model cache its own
#: sessions keep warm instead.
PROBE_BACKENDS = ACP_BACKENDS_ADVERTISED_MODEL_SELECTION - ACP_BACKENDS_ACP_RUNTIME

#: How long a successful catalog stays fresh. An account's entitlement changes
#: on a plan change, not on a page load, and the picker polls while degraded —
#: so without a TTL every poll would spawn an adapter.
CACHE_TTL_SECONDS = 900.0

#: How long a FAILED probe is remembered. Much shorter than a success: the
#: common failures (adapter not installed yet, not signed in) are exactly the
#: ones an operator fixes and then retries, and making them wait out the success
#: TTL would read as the fix not having worked. Long enough that a poll loop
#: cannot turn a broken adapter into a spawn storm.
FAILURE_TTL_SECONDS = 60.0


@dataclass
class AdapterCatalog:
    """What one handshake learned. Empty fields mean "not reported"."""

    models: list[dict[str, str]] = field(default_factory=list)
    effort_levels: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.models or self.effort_levels)


@dataclass
class _CacheEntry:
    catalog: AdapterCatalog
    expires_at: float


_cache: dict[str, _CacheEntry] = {}
# One lock per backend, so a probe of Claude does not serialize behind a probe
# of Codex, while concurrent pollers of the SAME backend collapse onto one
# spawn instead of starting one adapter each.
_locks: dict[str, asyncio.Lock] = {}


def _lock_for(backend: str) -> asyncio.Lock:
    lock = _locks.get(backend)
    if lock is None:
        lock = asyncio.Lock()
        _locks[backend] = lock
    return lock


def invalidate(backend: str | None = None) -> None:
    """Drop cached catalogs, so the next read re-probes.

    Called when the answer could have changed underneath the cache — an adapter
    installed, a harness switched — and by tests, which must not inherit another
    test's probe.
    """
    if backend is None:
        _cache.clear()
    else:
        _cache.pop(backend, None)


async def _handshake(backend: str, work_dir: Path, sandbox_mode: str) -> AdapterCatalog:
    """One spawn → ``session/new`` → read → teardown cycle."""
    # Imported here: acp.client imports this package's siblings at module level,
    # and a top-level import would re-enter that cycle.
    from kiro_crew.acp.client import AcpClient

    client = AcpClient(work_dir=work_dir, acp_backend=backend, sandbox_mode=sandbox_mode, model="")
    try:
        await asyncio.wait_for(client.ensure_ready(), timeout=PROBE_TIMEOUT_SECONDS)
        return AdapterCatalog(
            models=list(client.available_models()),
            effort_levels=list(client.get_valid_effort_levels()),
        )
    finally:
        # Always: a probe that timed out or failed mid-handshake still owns a
        # spawned adapter process, and leaking one per poll is worse than the
        # empty catalog the caller falls back to.
        try:
            await client.shutdown()
        except Exception:
            logger.debug("adapter catalog probe teardown failed", exc_info=True)


async def read_catalog(backend: str, work_dir: Path, sandbox_mode: str) -> AdapterCatalog:
    """Return *backend*'s advertised catalog, probing at most once per TTL.

    Never raises: a picker asking "what can this harness run" gets an empty
    answer and shows its ``auto`` sentinel, which is what it did before this
    probe existed. A failure is cached briefly so a degraded picker's poll loop
    cannot become a spawn loop.
    """
    if backend not in PROBE_BACKENDS:
        # Kiro and KAS answer from their own one-shot catalog command, and a
        # runtime-driven harness from its sessions' cache; see PROBE_BACKENDS.
        return AdapterCatalog()

    now = time.monotonic()
    entry = _cache.get(backend)
    if entry is not None and entry.expires_at > now:
        return entry.catalog

    async with _lock_for(backend):
        # Re-check under the lock: whoever held it may have just filled the
        # cache, and probing again would spawn a second adapter for nothing.
        entry = _cache.get(backend)
        now = time.monotonic()
        if entry is not None and entry.expires_at > now:
            return entry.catalog
        try:
            catalog = await _handshake(backend, work_dir, sandbox_mode)
        except Exception as exc:
            # Every failure mode lands here and is equivalent to the caller:
            # adapter missing, not signed in, handshake timeout, spawn refused.
            logger.info(
                "Adapter catalog probe failed for backend %r (%s); "
                "the picker falls back to its Auto sentinel",
                backend or "kiro",
                type(exc).__name__,
            )
            catalog = AdapterCatalog()
        ttl = CACHE_TTL_SECONDS if catalog else FAILURE_TTL_SECONDS
        _cache[backend] = _CacheEntry(catalog=catalog, expires_at=time.monotonic() + ttl)
        return catalog
