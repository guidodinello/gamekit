"""Round-robin league scheduler: every agent against every other, seat-rotated,
``n`` games per pairing, seeded and resumable one pairing at a time.

Each pairing is one ``gamekit.benchmark.run_arm`` call, so seat rotation and its
occupancy validation are inherited rather than re-implemented, and the result
is stamped with ``gamekit.results.stamp``. One JSON file per pairing is written
under ``results_dir / name /``; a re-run skips every pairing whose file already
exists *with a matching config hash* and refuses (``ValueError``) to reuse one
whose config differs.

**Multi-player games.** A pairing ``(a, b)`` in an ``N``-seat game fields the
alternating lineup ``(a, b, a, b, ...)``, so each agent holds ``N / 2`` seats
and the pairwise outcome is "which agent's seat won". Under a Luce
first-choice model, ``P(a wins) = k*pi_a / (k*pi_a + k*pi_b) = pi_a / (pi_a +
pi_b)`` -- exactly Bradley-Terry -- so two identical agents sit at 50% and get
equal ratings. A 1-vs-field lineup would put parity at ``1/N`` and misrate
identical agents, so it is not used. An odd ``N > 2`` cannot be balanced and
raises ``NotImplementedError``. Plackett-Luce over full rankings is future
work; see ``docs/research/022-league-ratings.md``.

**Roster independence.** Pairing seeds come from a BLAKE2b digest of ``(seed,
a, b)`` with ``a < b``, never from a pairing's index, so adding an agent later
leaves every existing pairing's seeds -- and therefore its resumable file --
untouched.

**Parallel pairings.** ``run_league`` plays pairings serially. ``run_pairing`` is
its loop body (hash check, ``run_arm``, atomic write) with no summary side effect,
so a consumer can run pairings from its own pool and then call
``summarize_league(load_pairings(league_dir), anchor=...)`` once. It writes the
same file ``run_league`` would, whatever the roster or argument order.

**Ties.** A game with no winning seat is dropped from the rating fit and the
win-rate denominators and reported per pairing as ``ties``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gamekit.benchmark import run_arm
from gamekit.league.ratings import (
    bootstrap_ratings,
    elo_ratings,
    find_cycles,
    fit_bradley_terry,
    win_matrix,
)
from gamekit.results import stamp, write_result
from gamekit.seats import rotate

_SEED_PREFIX = b"gamekit.league.pairing_seeds.v1\x00"
_PAIR_SEP = "__vs__"


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    """One game to play: its seeds and the already-rotated seat -> agent-name
    lineup. The game's ``play`` callable never re-implements rotation."""

    engine_seed: int
    driver_seed: int
    lineup: tuple[str, ...]


def pairing_lineup(a: str, b: str, num_seats: int) -> tuple[str, ...]:
    """The balanced alternating lineup ``(a, b, a, b, ...)`` for a pairing."""
    if num_seats < 2:
        raise ValueError(f"num_seats must be >= 2, got {num_seats}")
    if num_seats % 2:
        raise NotImplementedError(
            f"{num_seats} seats cannot be split evenly between two agents; only "
            "an even number of seats has a balanced pairwise reduction (see "
            "docs/research/022-league-ratings.md)"
        )
    return tuple(a if s % 2 == 0 else b for s in range(num_seats))


def pairing_seeds(seed: int, a: str, b: str) -> tuple[int, int]:
    """``(engine_seed_base, driver_seed_base)`` for the pairing ``{a, b}``.

    Order-insensitive in ``(a, b)`` and independent of the rest of the roster.
    """
    lo, hi = sorted((a, b))
    payload = _SEED_PREFIX + json.dumps([seed, lo, hi]).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    word = int.from_bytes(digest, "big")
    return word & 0x7FFFFFFF, (word >> 32) & 0x7FFFFFFF


def _config(a: str, b: str, num_seats: int, n: int, seed: int) -> dict[str, Any]:
    engine, driver = pairing_seeds(seed, a, b)
    return {
        "a": a,
        "b": b,
        "num_seats": num_seats,
        "n_games": n,
        "seed": seed,
        "engine_seed_base": engine,
        "driver_seed_base": driver,
    }


def _hash(config: Mapping[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _is_pairing_file(path: Path) -> bool:
    return _PAIR_SEP in path.name and not path.name.endswith(".partial.json")


def load_pairings(league_dir: Path) -> list[dict[str, Any]]:
    """Every finished pairing payload under ``league_dir``, sorted by name."""
    files = sorted(p for p in league_dir.glob("*.json") if _is_pairing_file(p))
    return [json.loads(p.read_text()) for p in files]


def _decisive(pairing: Mapping[str, Any]) -> dict[str, Any]:
    cfg = pairing["league_config"]
    a, b = cfg["a"], cfg["b"]
    wins_a = pairing["by_role"][a]["wins"]
    wins_b = pairing["by_role"][b]["wins"]
    return {
        "a": a,
        "b": b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": cfg["n_games"] - wins_a - wins_b,
    }


def summarize_league(
    pairings: Sequence[Mapping[str, Any]],
    *,
    anchor: str,
    anchor_rating: float = 0.0,
    prior_draws: float = 1.0,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Ratings, win matrix and cycle report from finished pairing payloads.

    Pure over the payloads, so it can be re-run from disk with a different
    anchor or prior and no game replayed.
    """
    counts: dict[tuple[str, str], float] = {}
    ties: dict[str, int] = {}
    agents: set[str] = set()
    for pairing in pairings:
        d = _decisive(pairing)
        agents.update((d["a"], d["b"]))
        counts[(d["a"], d["b"])] = d["wins_a"]
        counts[(d["b"], d["a"])] = d["wins_b"]
        ties[f"{d['a']}{_PAIR_SEP}{d['b']}"] = d["ties"]
    names = sorted(agents)
    if anchor not in agents:
        raise ValueError(f"anchor {anchor!r} is not one of the agents {names}")

    strengths = fit_bradley_terry(counts, names, prior_draws=prior_draws)
    elo = elo_ratings(strengths, anchor, anchor_rating)
    cis = bootstrap_ratings(
        counts,
        names,
        anchor=anchor,
        anchor_rating=anchor_rating,
        prior_draws=prior_draws,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
        alpha=alpha,
    )
    matrix = win_matrix(counts, names, strengths=strengths, alpha=alpha)
    return stamp(
        "league",
        anchor=anchor,
        anchor_rating=anchor_rating,
        prior_draws=prior_draws,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        alpha=alpha,
        agents=names,
        ratings={
            a: {"elo": elo[a], "elo_ci": list(cis[a])}
            for a in sorted(names, key=lambda x: -elo[x])
        },
        matrix={f"{i}{_PAIR_SEP}{j}": e for (i, j), e in matrix.items()},
        ties=ties,
        cycles=[list(c) for c in find_cycles(matrix, names)],
    )


def _validate_names(agents: Sequence[str]) -> None:
    for agent in agents:
        if "/" in agent or _PAIR_SEP in agent:
            raise ValueError(
                f"agent name {agent!r} may not contain '/' or {_PAIR_SEP!r}"
            )


def run_pairing[R](
    a: str,
    b: str,
    *,
    num_seats: int,
    n_per_pairing: int,
    seed: int,
    play: Callable[[Sequence[ScheduledGame]], Sequence[R]],
    winning_seat: Callable[[R], int | None],
    results_dir: Path,
    name: str,
    rotation_offset: Callable[[int], int] = lambda engine_seed: engine_seed,
) -> dict[str, Any]:
    """Play (or resume) the single pairing ``{a, b}`` and return its payload.

    This is ``run_league``'s loop body without the summary: it writes only
    ``results_dir/name/<lo>__vs__<hi>.json`` (atomically), never ``league.json``,
    and needs no anchor. ``(a, b)`` is order-insensitive, and the file, seeds
    and config hash equal what ``run_league`` writes, so pairings can be run
    from separate processes and summarized afterwards. A finished file with a
    matching config hash is returned without calling ``play``; a different hash
    raises ``ValueError``. Other arguments are as in ``run_league``.
    """
    if a == b:
        raise ValueError(f"a pairing needs two distinct agents, got {a!r} twice")
    _validate_names((a, b))
    a, b = sorted((a, b))
    lineup = pairing_lineup(a, b, num_seats)
    config = _config(a, b, num_seats, n_per_pairing, seed)
    chash = _hash(config)
    final = results_dir / name / f"{a}{_PAIR_SEP}{b}.json"
    if final.exists():
        stored: dict[str, Any] = json.loads(final.read_text())
        if stored.get("config_hash") != chash:
            raise ValueError(
                f"{final} was produced with a different config "
                f"(hash {stored.get('config_hash')}, expected {chash}); use a new "
                "name or results_dir instead of reusing it"
            )
        return stored

    def play_games(pairs: Sequence[tuple[int, int]]) -> Sequence[R]:
        return play(
            [
                ScheduledGame(
                    engine_seed=e,
                    driver_seed=d,
                    lineup=rotate(lineup, rotation_offset(e) % len(lineup)),
                )
                for e, d in pairs
            ]
        )

    payload = run_arm(
        lineup=lineup,
        num_seats=num_seats,
        n_games=n_per_pairing,
        engine_seed_base=config["engine_seed_base"],
        driver_seed_base=config["driver_seed_base"],
        play=play_games,
        winning_seat=winning_seat,
        rotation_offset=rotation_offset,
    )
    payload["league_config"] = config
    payload["config_hash"] = chash
    # write_result is write-then-rename, so a crash never leaves a half-written
    # pairing that a resume would trust.
    write_result(final.parent, final.stem, payload)
    return payload


def run_league[R](
    *,
    agents: Sequence[str],
    num_seats: int,
    n_per_pairing: int,
    seed: int,
    play: Callable[[Sequence[ScheduledGame]], Sequence[R]],
    winning_seat: Callable[[R], int | None],
    results_dir: Path,
    name: str,
    anchor: str,
    anchor_rating: float = 0.0,
    prior_draws: float = 1.0,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 0,
    rotation_offset: Callable[[int], int] = lambda engine_seed: engine_seed,
) -> dict[str, Any]:
    """Play (or resume) the round robin and return the league summary.

    ``play`` receives every ``ScheduledGame`` of one pairing (seeds plus the
    rotated lineup of agent names) and returns one result per game in order;
    ``winning_seat`` reads the winning seat from a result. Agent names are the
    identity: they must be distinct and may not contain ``/`` or ``__vs__``.
    ``n_per_pairing`` must be a multiple of ``num_seats`` (``run_arm``'s exact
    rotation contract). The summary is also written to
    ``results_dir/name/league.json`` (atomically). To play pairings in
    parallel without the summary, call ``run_pairing`` per pair instead.
    """
    if len(set(agents)) != len(agents) or len(agents) < 2:
        raise ValueError("agents must be at least two distinct names")
    _validate_names(agents)
    if anchor not in agents:
        raise ValueError(f"anchor {anchor!r} is not one of the agents")

    league_dir = results_dir / name
    for a, b in itertools.combinations(sorted(agents), 2):
        run_pairing(
            a,
            b,
            num_seats=num_seats,
            n_per_pairing=n_per_pairing,
            seed=seed,
            play=play,
            winning_seat=winning_seat,
            results_dir=results_dir,
            name=name,
            rotation_offset=rotation_offset,
        )

    # Only this run's roster: a directory reused after the roster shrank may
    # hold pairings (never config-checked here) for agents no longer in it.
    roster = set(agents)
    in_roster = [
        p
        for p in load_pairings(league_dir)
        if {p["league_config"]["a"], p["league_config"]["b"]} <= roster
    ]
    summary = summarize_league(
        in_roster,
        anchor=anchor,
        anchor_rating=anchor_rating,
        prior_draws=prior_draws,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    write_result(league_dir, "league", summary)
    return summary
