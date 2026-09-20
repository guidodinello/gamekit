# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking
- `gamekit.seats.seat_rng` now derives its seed from a BLAKE2b digest of
  `(driver_seed, seat)` instead of `hash((driver_seed, seat))`. This changes
  every stream `seat_rng` produces, so any committed result generated under
  the old stream is no longer reproducible from `seat_rng` alone -- either
  regenerate it, or replay it via `seat_rng_legacy`. See issue #8.

### Added
- `gamekit.seats.seat_rng_legacy` -- the pre-0.2.0 `hash()`-based stream,
  kept so already-committed results stay replayable.
- Initial project bootstrap with dev-standards baseline

### Deprecated
- `gamekit.seats.seat_rng_legacy` is deprecated on arrival: it exists only
  to replay pre-0.2.0 results and should not be used in new code.

## [0.1.0] - YYYY-MM-DD

### Added
- Project scaffolded