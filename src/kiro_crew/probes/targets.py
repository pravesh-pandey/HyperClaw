"""Infer WHICH subject a monitor instruction is about, from its own text.

The point of this module is that nothing new has to be passed in. A babysit
instruction already names its subject -- "Babysit PR #7491
(kirodotdev/KiroCrew, branch ...)" -- so asking the caller to also supply a
target parameter would add an opt-in, and an opt-in is what the previous
attempts at this saving died of: the parameter existed, nobody passed it, and
the measured adoption was zero. Inference has no adoption problem because there
is nothing to adopt.

The whole design leans on one asymmetry. Failing to infer costs a loop that
keeps its existing timer -- today's behaviour, no regression. Inferring the
WRONG subject costs a loop that watches something else: it goes quiet about the
thing it was supposed to watch and wakes about a stranger. So every rule here
refuses on doubt, and the refusal path is the tested one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from kiro_crew.probes import GH_PR

#: ``https://github.com/owner/name/pull/123`` (any host path prefix is refused
#: by the anchor -- an enterprise host is a different API and a different probe).
_PR_URL = re.compile(
    r"https?://(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9._-]+)/(?P<repo>[A-Za-z0-9._-]+)/pull/(?P<pr>\d+)\b"
)

#: ``owner/name#123``, the shorthand a babysit instruction usually carries.
#:
#: The lookbehind refuses a PATH fragment. A babysit instruction routinely cites
#: source locations, and ``src/kiro_crew/autonudge.py#1751`` would otherwise read
#: as owner ``kiro_crew`` / repo ``autonudge.py`` / PR 1751 -- a target that does
#: not exist, on a repository that does not exist, silencing the loop about the
#: PR it owns. A repository name may legitimately contain a dot (``foo.js``), so
#: the filename shape cannot be excluded by its suffix; what distinguishes the
#: two is that a path fragment has more path to its left.
_PR_SHORTHAND = re.compile(
    r"(?<![A-Za-z0-9._/#-])" r"(?P<owner>[A-Za-z0-9._-]+)/(?P<repo>[A-Za-z0-9._-]+)#(?P<pr>\d+)\b"
)


@dataclass(frozen=True)
class Target:
    """One inferred subject, ready to hand to a driver."""

    kind: str
    #: Human identity, for logs and for the loop's own bookkeeping.
    subject: str
    #: The probe's configuration, in the shape the probe already parses.
    message: str


def infer(text: str) -> Target | None:
    """Return the single subject *text* is about, or ``None``.

    ``None`` on every doubtful case, and specifically when the text names more
    than one distinct pull request. That case is common and it is exactly where
    guessing does damage: a babysit instruction routinely names its own PR *and*
    a PR it is blocked on ("gated on #4137 merging first"), and a watch armed on
    the blocker would report the blocker's progress while staying silent about
    the PR the loop actually owns.
    """
    if not isinstance(text, str) or not text:
        return None

    found: set[tuple[str, str, int]] = set()
    for pattern in (_PR_URL, _PR_SHORTHAND):
        for match in pattern.finditer(text):
            number = int(match.group("pr"))
            if number <= 0:
                continue
            found.add((match.group("owner"), match.group("repo"), number))

    # Exactly one subject, or nothing. Ambiguity is not resolved by preferring
    # the first mention: reading order does not tell which PR the loop owns, and
    # a rule that looks like it decides is worse than one that declines.
    if len(found) != 1:
        return None

    owner, repo, number = found.pop()
    slug = f"{owner}/{repo}"
    return Target(
        kind=GH_PR,
        subject=f"{slug}#{number}",
        # known_reds is deliberately absent: inference cannot know which reds
        # are inherited from the base branch, and inventing that list would
        # either suppress a real failure or wake on a known one. The woken agent
        # is where that judgment already lives.
        #
        # ``host`` is pinned because everything this function yields came from a
        # public GitHub spelling -- an enterprise URL is refused above, so there
        # is no case where the subject is somewhere else. Without the pin the
        # probe would address the subject as a bare ``owner/name`` slug and an
        # ambient ``GH_HOST`` would resolve it against a different server, where a
        # same-numbered pull request could be merged and stop a watch on a live
        # one.
        message=json.dumps({"repo": slug, "pr": number, "host": "github.com"}),
    )
