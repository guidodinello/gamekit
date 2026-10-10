"""Import-check an *installed* gamekit wheel (run from outside the checkout).

Default mode imports every module except ``gamekit.rl.env`` and asserts that
numpy/gymnasium were not pulled in: the core is stdlib-only and a plain
``pip install gamekit`` must not need them. ``--rl`` additionally imports
``gamekit.rl.env`` and asserts gymnasium is then loaded (the [rl] extra works).

Run it in a venv where the wheel is installed, with ``python -I`` and a cwd
outside the repo, so the checkout's ``src/`` cannot shadow the wheel.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

import gamekit

ENV_MODULE = "gamekit.rl.env"
FORBIDDEN = ("numpy", "gymnasium")


def _reraise(name: str) -> None:
    raise


def main() -> int:
    rl = "--rl" in sys.argv[1:]

    origin = Path(gamekit.__file__).resolve()
    if "site-packages" not in origin.parts:
        print(f"FAIL: gamekit imported from {origin}, not an installed wheel")
        return 1

    names = sorted(
        m.name
        for m in pkgutil.walk_packages(
            gamekit.__path__, prefix="gamekit.", onerror=_reraise
        )
    )
    if ENV_MODULE not in names:
        print(f"FAIL: {ENV_MODULE} missing from the wheel (found: {names})")
        return 1

    core = [n for n in names if n != ENV_MODULE]
    for name in core:
        importlib.import_module(name)
    leaked = [m for m in FORBIDDEN if m in sys.modules]
    if leaked:
        print(f"FAIL: core modules imported {leaked}")
        return 1
    print(f"ok: imported {len(core)} core modules; no {'/'.join(FORBIDDEN)}")

    if rl:
        importlib.import_module(ENV_MODULE)
        if "gymnasium" not in sys.modules:
            print(f"FAIL: {ENV_MODULE} imported without gymnasium")
            return 1
        print(f"ok: {ENV_MODULE} imports with the [rl] extra")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
