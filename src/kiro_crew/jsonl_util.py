"""Shared bounding helpers for append-only JSONL logs — on write and on read.

Several long-lived JSONL logs (the MCP stub's fallback audit log, the
subagents' slow-command log, per-member activity logs) grow one appended
record at a time from writers that must never stall an event loop. Each
needs the same bound: once the live file reaches its cap, rename it to a
single ``.1`` generation — O(1), no whole-file read — so total disk stays
at roughly twice the cap while one generation of history is kept.

:func:`rotate_jsonl_at` owns that rotation step. Each call site keeps its
own append, record shape, size cap, and error contract, because those
differ per log; what they share is exactly the rotate-by-rename.

:func:`bounded_records` and :func:`strict_records` own the matching bound on
the READ side. A file's total size being rotated does not bound one RECORD:
``for line in handle`` asks for bytes up to the next newline, so a single
crafted newline-free line is one allocation the size of the whole file, and
every tree these readers touch is agent-writable. The two functions differ
only in the posture an over-cap record takes, which is a per-call-site
judgement and the reason both exist:

* :func:`bounded_records` SKIPS the record. Correct where the read is
  read-only and degradable — the caller already tolerates a malformed
  record, and dropping one costs a count, not durable state.
* :func:`strict_records` ABORTS the read. Correct where the parsed output
  feeds a rewrite or any other durable decision, because a silently
  skipped record there loses or duplicates data.

The two postures differ in more than the over-cap branch, and that is the
point rather than an accident. A skip reader only has to be good enough to
count with, so it decodes with ``errors="replace"`` and accepts that a bare
CR no longer ends a record. A strict reader's caller PERSISTS what it read,
so neither is acceptable there: a replacement character would be written
back in place of the original bytes, and two CR-glued records would hide a
prior entry from a dedupe probe. The strict readers therefore make one
guarantee — a record they yield is exactly the record on disk — and raise
:class:`UnreadableRecord` for every other outcome.

Each has an undecoded twin — :func:`bounded_raw_records` and
:func:`strict_raw_records` — yielding ``bytes``, for the two callers that
must not go through a decode: one already iterated a binary handle and
prefilters on bytes before parsing, and one copies records into another file
verbatim, where a lossy ``errors="replace"`` decode and re-encode would
rewrite an undecodable byte as U+FFFD.

:func:`kiro_crew.session_storage._manifest_records` is a third reader and
deliberately stays separate: it needs ``str.splitlines`` boundaries, since
it replaced a whole-file ``splitlines`` and a manifest split on any unicode
line boundary must keep parsing as it always did. The two readers here use
``readline`` boundaries, which is what the handle iteration they replace
used.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import IO

from kiro_crew import platform_compat

logger = logging.getLogger(__name__)

# Longest record these readers will materialise, in BYTES (terminator excluded:
# the threshold applies to the record without its newline, so a cap-length record
# plus its terminator is accepted and the peak read is cap+1).
#
# The bound is in bytes, not characters, and callers open their file binary for
# that reason: a character cap is not a memory bound, because one astral code
# point is four bytes of `str` under PEP 393, so a 128 MiB CHARACTER cap admits
# half a gibibyte of resident text. Reading N bytes bounds the decoded `str` at a
# small constant multiple of N -- NOT at N exactly, because `errors="replace"`
# turns each undecodable byte into a U+FFFD costing 2 or 4 bytes of `str`
# depending on the widest code point in the record. That is still a bound flat in
# FILE size, which is the property the cap exists for, and it is why the peak is
# a constant multiple of this value rather than equal to it.
#
# One cap serves every caller, and it is sized for the LARGEST record shape any
# of them reads: a session transcript record, which carries a whole conversation
# turn. Its legitimate ceiling is ~90 MB -- image_artifacts'
# ``MAX_IMAGE_BYTES_PER_MESSAGE`` (64 MiB) base64-expands to ~85 MB, plus text --
# and the largest observed on a live install (30,351 kiro-cli logs, 27 GB) is
# 77,920,032 bytes, itself load-bearing: it carries an image content item, and the
# largest ``Prompt`` record (turns and first_message) is 56,203,168 bytes. 128 MiB
# is the smallest round value that clears that ceiling with margin.
#
# The other shapes read through here -- token-usage rows, telemetry export cycles,
# notification records, member activity entries (~150 bytes), stub fallback rows,
# cost samples -- are orders of magnitude smaller, so this cap never undercounts
# them either. A per-format cap would bound each tighter, but the memory that
# matters is the peak of one record, and a cap set below a format's real ceiling
# buys nothing while risking a silent undercount of legitimate data. This is why
# the cap is NOT session_storage's ``_MANIFEST_RECORD_CAP`` (8 MiB): a manifest
# record is a header or one session's file list, and 8 MiB here would silently
# truncate the biggest real sessions.
RECORD_CAP = 128 * 1024 * 1024


class UnreadableRecord(Exception):
    """A record cannot be delivered INTACT, so the file cannot be read completely.

    Raised only by the strict readers. The abort posture exists because its
    callers write based on what they parsed, so a record they cannot reproduce
    faithfully is not something to paper over -- every way of failing to
    deliver one intact has to reach the caller, not just an over-cap one.
    Callers map this to their own fail-closed posture and must not treat it
    as "no more records".
    """


class OversizedRecord(UnreadableRecord):
    """A record exceeded the cap, so it was never materialised."""


class UndecodableRecord(UnreadableRecord):
    """A record is not valid UTF-8, so decoding it would alter its bytes.

    The skip readers decode with ``errors="replace"``, which is right for a
    caller that only counts or displays: one mangled character costs it a
    record's contribution. It is wrong for a caller that writes what it read,
    because the replacement is what gets persisted -- ``compact_cost_log``
    would ``os.replace`` the log with U+FFFD substituted for the original
    bytes, and a dedupe key built from a replaced record no longer matches the
    record it came from.
    """


class AmbiguousRecordBoundary(UnreadableRecord):
    """A record body contains a bare CR, which used to end a record here.

    These files were read in TEXT mode before, where universal-newline
    translation made a lone ``\\r`` a record terminator. Binary reads split
    only on ``\\n``, so two CR-delimited records arrive glued into one. The
    skip readers accept that narrowing -- the glued line fails to parse and
    costs those records' contribution, which is their documented posture. A
    strict caller cannot: at :func:`kiro_crew.members.record_activity` the
    dedupe probe would stop seeing a prior entry and append a duplicate. No
    writer of these files emits a bare CR (``json.dumps`` escapes one inside a
    string as two ASCII characters), so refusing is free for real data and
    fails closed on a planted file.
    """


def _raw_records(handle: IO[bytes], cap: int, *, drain: bool) -> Iterator[bytes | None]:
    """Yield each record's raw bytes, or ``None`` for one over the cap.

    Reading ``cap + 1`` makes the verdict unambiguous: a returned chunk
    shorter than that either ended on a newline or hit EOF, so it is a whole
    record. Nothing larger is ever held.

    *drain* decides what happens after an over-cap record. When true the
    rest of that record is read and discarded, so the following record
    starts on a real boundary — without it, the next bounded read would
    begin mid-record and a hostile line's tail could forge a well-formed
    record that the reader had just reported as skipped. When false the
    generator stops instead, because its caller is aborting and reading a
    multi-GB tail it will discard would only donate the time.
    """
    while True:
        raw = handle.readline(cap + 1)
        if not raw:
            return
        if len(raw) > cap and not raw.endswith(b"\n"):
            if not drain:
                yield None
                return
            while True:
                tail = handle.readline(cap + 1)
                if not tail or tail.endswith(b"\n"):
                    break
            yield None
            continue
        yield raw


def _decode(raw: bytes) -> str:
    """Decode one accepted record.

    Per-record decoding is equivalent to decoding the whole file, because
    ``\\n`` is never part of a UTF-8 multi-byte sequence — lead and
    continuation bytes are all >= 0x80 — so no sequence can straddle a
    record boundary and be replaced differently than it would have been.
    """
    return raw.decode("utf-8", errors="replace")


def bounded_raw_records(
    handle: IO[bytes], path: Path, *, cap: int = RECORD_CAP, label: str = "read"
) -> Iterator[bytes]:
    """Yield *handle*'s records as raw bytes, SKIPPING any over *cap* bytes.

    The undecoded form of :func:`bounded_records`, for a caller that already
    iterated a binary handle and prefilters on bytes before parsing. Such a
    caller gains the bound with no other change; routing it through the
    decoding variant instead would make it decode every record to run a
    filter that rejects most of them.

    The skip posture. Use where the read is read-only and degradable: the
    caller already skips a record it cannot parse, so an over-cap record
    costs that one record's contribution and nothing durable. Where the
    output instead feeds a rewrite or a durable decision, use
    :func:`strict_records`.

    *handle* is BINARY so the cap is a memory bound rather than a code-point
    count (see :data:`RECORD_CAP`). Record boundaries are ``readline``'s,
    which is what the ``for line in handle`` iteration this replaces used, so
    a record containing an exotic line boundary (``\\u2028``, ``\\x1c``, ...)
    stays one record and still parses. For a caller converted from TEXT mode,
    binary reads narrow that in one direction: universal-newline translation
    is gone, so a lone ``\\r`` no longer ends a record and two CR-delimited
    records arrive glued into one. No writer of these files emits a bare CR --
    ``json.dumps`` escapes one inside a string as two ASCII characters -- so
    on real data this changes nothing, and on a planted file the glued line
    fails to parse and costs those records' contribution, which is exactly
    this reader's posture. A caller that cannot afford that must use
    :func:`strict_records`, which refuses the record instead (see
    :class:`AmbiguousRecordBoundary`). A ``\\r\\n`` terminator survives,
    because every caller strips the record before parsing -- it only shifts
    the boundary by one byte, since the ``\\r`` counts toward the read, so a
    ``\\r\\n``-terminated record is refused one byte sooner than an
    ``\\n``-terminated one. Immaterial at this cap; stated so a future reader
    does not read the threshold as byte-exact on a CRLF file.

    An over-cap record's tail is drained without being kept, so a hostile
    file costs time proportional to its length and memory proportional to
    the cap. *label* names the caller in the one debug line emitted when
    anything was skipped; that line runs only if the generator is driven to
    exhaustion, and a caller that stops early forfeits it knowingly.
    """
    oversized = 0
    for raw in _raw_records(handle, cap, drain=True):
        if raw is None:
            oversized += 1
            continue
        yield raw
    if oversized:
        # %r: *path* names a file in an agent-writable tree, and several
        # callers discover their inputs with iterdir()/glob(), so a planted
        # name can embed a newline. The repr keeps one log record from
        # forging others.
        logger.debug("%s: skipped %d record(s) over %d bytes in %r", label, oversized, cap, path)


def bounded_records(
    handle: IO[bytes], path: Path, *, cap: int = RECORD_CAP, label: str = "read"
) -> Iterator[str]:
    """Yield *handle*'s records decoded, SKIPPING any over *cap* bytes.

    :func:`bounded_raw_records` plus :func:`_decode`, and the form almost
    every caller wants: it replaces a text-mode ``for line in handle``
    without touching the loop body. See that function for the posture,
    boundary and cap properties they share.
    """
    for raw in bounded_raw_records(handle, path, cap=cap, label=label):
        yield _decode(raw)


def _refuse_bare_cr(raw: bytes, path: Path) -> None:
    """Raise if *raw*'s body holds a CR that a text-mode read would have split on.

    A trailing ``\\r\\n`` is fine: that CR rides on the terminator and every
    caller strips the record before parsing. A CR anywhere else WAS a record
    boundary before these files moved to binary reads, so accepting it would
    silently glue two records into one. See :class:`AmbiguousRecordBoundary`.
    """
    body = raw[:-2] if raw.endswith(b"\r\n") else (raw[:-1] if raw.endswith(b"\n") else raw)
    if b"\r" in body:
        raise AmbiguousRecordBoundary(f"bare CR inside a record in {path!r}")


def strict_raw_records(handle: IO[bytes], path: Path, *, cap: int = RECORD_CAP) -> Iterator[bytes]:
    """Yield *handle*'s records as raw bytes, raising on any it cannot deliver intact.

    The abort posture, undecoded. Use where the record must survive
    byte-for-byte: a reader that copies records into another file cannot go
    through a lossy ``errors="replace"`` decode and re-encode, which would
    rewrite an undecodable byte as U+FFFD.

    Raises :class:`OversizedRecord` past *cap* and
    :class:`AmbiguousRecordBoundary` on a bare CR -- both are
    :class:`UnreadableRecord`, which is what callers should catch. Nothing
    past ``cap + 1`` bytes is ever held, and an over-cap record's tail is NOT
    drained: the caller is abandoning the read, so walking a multi-GB line
    to its end would hand the hostile file the very cost the cap exists to
    deny it.
    """
    for raw in _raw_records(handle, cap, drain=False):
        if raw is None:
            raise OversizedRecord(f"record over {cap} bytes in {path!r}")
        _refuse_bare_cr(raw, path)
        yield raw


def strict_records(handle: IO[bytes], path: Path, *, cap: int = RECORD_CAP) -> Iterator[str]:
    """Yield *handle*'s records decoded STRICTLY, raising on any it cannot deliver intact.

    The abort posture. Use where the parsed output feeds a rewrite or another
    durable decision: skipping a record there would silently drop it from
    what gets written back, or hide it from a dedupe probe so a duplicate
    is appended.

    Unlike :func:`bounded_records` this decodes with ``errors="strict"`` and
    turns a failure into :class:`UndecodableRecord`, because here the
    replacement character is what the caller PERSISTS rather than a display
    artefact. With :class:`OversizedRecord` and
    :class:`AmbiguousRecordBoundary` that adds up to one guarantee worth
    stating plainly: a record this generator yields is exactly the record on
    disk, and every other outcome stops the read. Catch
    :class:`UnreadableRecord` to cover all three.

    See :func:`strict_raw_records` for the no-drain rationale.
    """
    for raw in strict_raw_records(handle, path, cap=cap):
        try:
            yield raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UndecodableRecord(f"record is not valid UTF-8 in {path!r}") from exc


def rotate_jsonl_at(path: Path, max_bytes: int) -> None:
    """Rotate ``path`` aside to ``<name>.1`` once it reaches ``max_bytes``.

    Call immediately before appending a record. Keeps ONE previous
    generation, replacing any older one, so total disk use stays bounded at
    about twice the cap. The live file can overshoot the cap by the few
    records written between a size check and the next rotation; callers
    accept that slack in exchange for never blocking.

    Rotation is guarded by a NON-BLOCKING try-lock on a sibling
    ``<name>.lock`` file so that two writers hitting the cap together
    cannot both rotate (the second would replace ``.1`` with the first's
    fresh live file, discarding a generation). A loser skips rotating — it
    never waits, so no caller can stall its event loop — and the next
    writer rotates. Every current caller is (or must be treated as) a
    multi-process writer, so the lock is unconditional; the cost to a
    single writer is one fd and one non-blocking syscall.

    Best-effort by contract: NEVER raises. Any failure — the lock file
    unopenable (fd exhaustion, read-only or ACL-restricted dir), a
    fresh-boot missing log, a Windows sharing violation rejecting the
    rename, an unusable path value — degrades to not rotating, so the
    caller's append still runs. Fd/disk exhaustion is a leading cause of
    the very incidents these logs diagnose, so a rotation failure must
    never cost the record; only a failure of the caller's own append may.
    """
    try:
        lock_fd = os.open(path.with_name(path.name + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            locked = platform_compat.try_acquire_lock(lock_fd, exclusive=True)
            try:
                if locked and path.stat().st_size >= max_bytes:
                    os.replace(path, path.with_name(path.name + ".1"))
            finally:
                if locked:
                    platform_compat.release_lock(lock_fd)
        finally:
            os.close(lock_fd)
    except (OSError, ValueError):
        pass
