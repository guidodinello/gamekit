"""Bradley-Terry ratings over a pairwise win table: pure statistics, no I/O.

Stdlib-only on purpose. ``pyproject.toml`` keeps every core module free of
numpy (it arrives only with the ``[rl]`` extra), and the CI legs that run
without that extra would fail on a top-level numpy import. Hunter's MM
iteration is a few lines of plain Python and is fast for the tens of agents a
league has.

**Model.** Agent ``i`` has strength ``pi_i > 0`` and beats ``j`` with
probability ``pi_i / (pi_i + pi_j)`` (Bradley-Terry). The MLE is found with the
MM update ``pi_i <- W_i / sum_j n_ij / (pi_i + pi_j)`` (Hunter 2004), where
``W_i`` is ``i``'s total wins and ``n_ij`` the decisive games of the pairing.

**Separation prior.** An agent that wins (or loses) every game has no finite
MLE. Every fit therefore adds ``prior_draws`` virtual games per played pairing,
split evenly -- a weak regularizer in the spirit of BayesElo's prior, not a
reimplementation of it; ``prior_draws=0`` is the plain MLE and raises
``ValueError`` when the win graph is not strongly connected.

**Anchoring.** Strengths are only defined up to a common scale, so ratings are
reported on the Elo scale (``400 * log10(pi_i / pi_anchor)``) relative to a
named anchor agent, shifted by ``anchor_rating``. The anchor's own bootstrap
interval is therefore exactly ``[anchor_rating, anchor_rating]``; every other
interval is uncertainty *relative to the anchor*.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from gamekit.mc import ConfidenceInterval, wilson_interval

# (i, j) -> number of decisive games i won against j.
type PairwiseCounts = Mapping[tuple[str, str], float]

_ELO_PER_LOGIT = 400 / math.log(10)


def _games(counts: PairwiseCounts, i: str, j: str) -> float:
    return counts.get((i, j), 0) + counts.get((j, i), 0)


def _played_pairs(counts: PairwiseCounts) -> list[tuple[str, str]]:
    """Each unordered pairing with at least one game, as a sorted pair."""
    pairs = {(min(i, j), max(i, j)) for (i, j), w in counts.items() if i != j and w > 0}
    return sorted(pairs)


def _with_prior(
    counts: PairwiseCounts, prior_draws: float
) -> dict[tuple[str, str], float]:
    out = {k: float(v) for k, v in counts.items()}
    if prior_draws > 0:
        for i, j in _played_pairs(counts):
            out[(i, j)] = out.get((i, j), 0.0) + prior_draws / 2
            out[(j, i)] = out.get((j, i), 0.0) + prior_draws / 2
    return out


def _reachable(start: str, edges: Mapping[str, set[str]]) -> set[str]:
    seen = {start}
    stack = [start]
    while stack:
        for nxt in edges.get(stack.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _check_connected(
    counts: PairwiseCounts, agents: Sequence[str], *, strongly: bool
) -> None:
    fwd: dict[str, set[str]] = {a: set() for a in agents}
    bwd: dict[str, set[str]] = {a: set() for a in agents}
    for (i, j), w in counts.items():
        if w > 0:
            fwd[i].add(j)
            bwd[j].add(i)
            if not strongly:
                fwd[j].add(i)
                bwd[i].add(j)
    first = agents[0]
    if len(_reachable(first, fwd)) != len(agents) or (
        len(_reachable(first, bwd)) != len(agents)
    ):
        kind = (
            "strongly connected (every agent must both win and lose)"
            if (strongly)
            else "connected (some agents never played the others)"
        )
        raise ValueError(f"comparison graph is not {kind}; ratings are undefined")


def fit_bradley_terry(
    counts: PairwiseCounts,
    agents: Sequence[str],
    *,
    prior_draws: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 10_000,
    init: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """MLE log-strengths (natural log, mean zero) by Hunter's MM iteration.

    ``counts[(i, j)]`` is the number of decisive games ``i`` won against
    ``j``. ``init`` warm-starts from known log-strengths (used by the
    bootstrap).
    """
    if prior_draws < 0:
        raise ValueError("prior_draws must be >= 0")
    if len(set(agents)) != len(agents) or len(agents) < 2:
        raise ValueError("need at least two distinct agents")
    data = _with_prior(counts, prior_draws)
    _check_connected(data, agents, strongly=prior_draws == 0)
    _check_connected(data, agents, strongly=False)

    wins = {a: sum(w for (i, _), w in data.items() if i == a) for a in agents}
    opponents = {
        a: [b for b in agents if b != a and _games(data, a, b) > 0] for a in agents
    }
    theta = {a: (init or {}).get(a, 0.0) for a in agents}
    pi = {a: math.exp(theta[a]) for a in agents}
    for _ in range(max_iter):
        new = {}
        for a in agents:
            denom = sum(_games(data, a, b) / (pi[a] + pi[b]) for b in opponents[a])
            new[a] = wins[a] / denom
        log_mean = sum(math.log(v) for v in new.values()) / len(new)
        scale = math.exp(log_mean)
        new = {a: v / scale for a, v in new.items()}
        delta = max(abs(math.log(new[a]) - math.log(pi[a])) for a in agents)
        pi = new
        if delta < tol:
            break
    return {a: math.log(pi[a]) for a in agents}


def elo_ratings(
    strengths: Mapping[str, float], anchor: str, anchor_rating: float = 0.0
) -> dict[str, float]:
    """Elo-scale ratings from log-strengths, with ``anchor`` pinned to
    ``anchor_rating``."""
    if anchor not in strengths:
        raise ValueError(f"anchor {anchor!r} is not one of the agents")
    base = strengths[anchor]
    return {
        a: anchor_rating + _ELO_PER_LOGIT * (s - base) for a, s in strengths.items()
    }


def predicted_win_rate(strengths: Mapping[str, float], i: str, j: str) -> float:
    """BT probability that ``i`` beats ``j``."""
    return 1 / (1 + math.exp(strengths[j] - strengths[i]))


def bootstrap_ratings(
    counts: PairwiseCounts,
    agents: Sequence[str],
    *,
    anchor: str,
    anchor_rating: float = 0.0,
    prior_draws: float = 1.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, ConfidenceInterval]:
    """Percentile bootstrap CIs on the anchored Elo ratings.

    Stratified by pairing: each replicate redraws every pairing's decisive
    wins as ``Binomial(n_ij, observed rate)`` -- equivalent to resampling
    games within a pairing, from the prior-smoothed rate -- then refits (with
    the same prior) and re-anchors.
    """
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be >= 2")
    rng = random.Random(seed)
    pairs = _played_pairs(counts)
    point = fit_bradley_terry(counts, agents, prior_draws=prior_draws)
    draws: dict[str, list[float]] = {a: [] for a in agents}
    for _ in range(n_bootstrap):
        resampled: dict[tuple[str, str], float] = {}
        for i, j in pairs:
            n = round(_games(counts, i, j))
            # Smoothed with the same prior as the fit: at a raw 10-0 the plain
            # rate is 1.0 and every replicate would be 10-0, a falsely
            # zero-width interval.
            p = (counts.get((i, j), 0) + prior_draws / 2) / (n + prior_draws)
            w = rng.binomialvariate(n, p)
            resampled[(i, j)] = w
            resampled[(j, i)] = n - w
        strengths = fit_bradley_terry(
            resampled, agents, prior_draws=prior_draws, init=point
        )
        for a, r in elo_ratings(strengths, anchor, anchor_rating).items():
            draws[a].append(r)
    out: dict[str, ConfidenceInterval] = {}
    for a, xs in draws.items():
        xs.sort()
        lo = xs[int((alpha / 2) * (len(xs) - 1))]
        hi = xs[math.ceil((1 - alpha / 2) * (len(xs) - 1))]
        out[a] = ConfidenceInterval(lo, hi)
    return out


def win_matrix(
    counts: PairwiseCounts,
    agents: Sequence[str],
    *,
    strengths: Mapping[str, float] | None = None,
    alpha: float = 0.05,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Raw pairwise table for every played ordered pair ``(i, j)``: decisive
    wins/losses, win rate with a Wilson CI, and (given ``strengths``) the BT
    prediction and the residual ``observed - predicted``."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for i in agents:
        for j in agents:
            if i == j:
                continue
            w, lost = counts.get((i, j), 0), counts.get((j, i), 0)
            n = round(w + lost)
            if n == 0:
                continue
            rate = w / n
            ci = wilson_interval(round(w), n, alpha)
            entry: dict[str, Any] = {
                "wins": round(w),
                "losses": round(lost),
                "n_decisive": n,
                "win_rate": rate,
                "win_rate_wilson_ci": list(ci),
            }
            if strengths is not None:
                pred = predicted_win_rate(strengths, i, j)
                entry["bt_predicted"] = pred
                entry["residual"] = rate - pred
            out[(i, j)] = entry
    return out


def find_cycles(
    matrix: Mapping[tuple[str, str], Mapping[str, Any]],
    agents: Iterable[str],
) -> list[tuple[str, str, str]]:
    """Every 3-cycle ``A > B > C > A`` among *significant* edges.

    ``A -> B`` is an edge only when the Wilson CI of A's decisive win rate
    against B lies entirely above 0.5; otherwise noise alone would manufacture
    cycles. Each cycle is reported once, rotated to start at its smallest name.
    """
    names = sorted(agents)
    beats = {(i, j) for (i, j), e in matrix.items() if e["win_rate_wilson_ci"][0] > 0.5}
    found: list[tuple[str, str, str]] = []
    for a in names:
        for b in names:
            if b <= a or (a, b) not in beats:
                continue
            for c in names:
                if c <= a or c == b:
                    continue
                if (b, c) in beats and (c, a) in beats:
                    found.append((a, b, c))
    return found
