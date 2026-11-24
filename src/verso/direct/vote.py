"""Votes among several direct solutions of one problem (decision D75): the answers grouped when they agree, the
largest group first.

Two numeric answers agree within RELATIVE_TOLERANCE of the larger (the grader's 1%); several roots agree when they
agree one by one; "no solution" agrees with "no solution". A solution without a gradable answer casts no vote.
"""

from ..core.types import SolveResult, SolveStatus

RELATIVE_TOLERANCE = 0.01
_ANSWERED = (SolveStatus.OK, SolveStatus.MULTIPLE, SolveStatus.NO_SOLUTION)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= RELATIVE_TOLERANCE * max(abs(a), abs(b)) or abs(a - b) <= 1e-12


def agree(a: SolveResult, b: SolveResult) -> bool:
    """Whether two answers are the same answer."""
    if a.status != b.status or a.status not in _ANSWERED:
        return False
    if a.status == SolveStatus.NO_SOLUTION:
        return True
    if a.status == SolveStatus.MULTIPLE:
        return len(a.values) == len(b.values) and all(_close(x, y) for x, y in zip(sorted(a.values), sorted(b.values)))
    return a.value is not None and b.value is not None and _close(a.value, b.value)


def groups(results: list[SolveResult]) -> list[tuple[SolveResult, int]]:
    """(answer, votes) of each group of agreeing answers, by votes, then by first appearance; the answer of a group
    is its first one."""
    found: list[list] = []
    for r in results:
        if r is None or r.status not in _ANSWERED:
            continue
        for g in found:
            if agree(g[0], r):
                g[1] += 1
                break
        else:
            found.append([r, 1])
    order = sorted(range(len(found)), key=lambda i: (-found[i][1], i))
    return [(found[i][0], found[i][1]) for i in order]
