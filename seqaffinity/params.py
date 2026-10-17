"""Defaults, the TRAP R0 formula and parameter validation."""

from __future__ import annotations

import math
import warnings

DEFAULT_LAMBDA = 0.7
DEFAULT_WINDOW = 200
DEFAULT_STEP = 50
DEFAULT_CHUNK_SIZE = 10_000_000
DEFAULT_MIN_VALID_FRACTION = 0.5
DEFAULT_PROBABILITY_DEPTH = 100.0
R0_CAP = 10.0

#: Canonical score keys, in output order.
SCORE_KEYS = ("max", "tba", "tba_per_window", "trap", "pf")

#: Output column(s) produced by each score key.
SCORE_COLUMNS = {
    "max": ("max_score",),
    "tba": ("ln_tba",),
    "tba_per_window": ("tba_per_window",),
    "trap": ("trap", "trap_per_bp"),
    "pf": ("partition_fn",),
}


def trap_r0(w: int, cap: float = R0_CAP, warn: bool = True) -> float:
    """TRAP's empirical effective concentration for a motif of width ``w``.

    ``ln R0 = 0.584·w − 5.66`` (Roider et al. 2007), capped at ``cap``.

    Parameters
    ----------
    w : int
        Motif width.
    cap : float
        Upper bound on R0. The default cap of 10 engages at ``w >= 14``.
    warn : bool
        Emit a warning when the cap engages.

    Returns
    -------
    float
    """
    r0 = math.exp(0.584 * w - 5.66)
    if r0 > cap:
        if warn:
            warnings.warn(
                f"TRAP R0 formula gives {r0:.3g} for w={w}; capped at {cap:g}. "
                "Results will differ from tRap for wide motifs.",
                stacklevel=2,
            )
        return float(cap)
    return r0


def normalise_scores(scores) -> tuple[str, ...]:
    """Validate a score selection and return it in canonical order."""
    if isinstance(scores, str):
        scores = [s.strip() for s in scores.split(",") if s.strip()]
    scores = set(scores)
    unknown = scores - set(SCORE_KEYS)
    if unknown:
        raise ValueError(f"unknown score(s) {sorted(unknown)}; choose from {SCORE_KEYS}")
    if not scores:
        raise ValueError("at least one score must be selected")
    return tuple(k for k in SCORE_KEYS if k in scores)


def score_columns(scores) -> list[str]:
    """Output column names for a (normalised) score selection."""
    return [c for k in scores for c in SCORE_COLUMNS[k]]


def validate_lambda(lam: float) -> float:
    lam = float(lam)
    if not (lam > 0 and math.isfinite(lam)):
        raise ValueError(f"lambda must be positive and finite, got {lam}")
    return lam


def validate_r0(r0: float) -> float:
    r0 = float(r0)
    if not (r0 > 0 and math.isfinite(r0)):
        raise ValueError(f"R0 must be positive and finite, got {r0}")
    return r0


def validate_strands(strands: str) -> str:
    if strands not in ("both", "plus"):
        raise ValueError(f"strands must be 'both' or 'plus', got {strands!r}")
    return strands


def validate_scan(W: int, s: int, w: int, min_valid_fraction: float, chunk_size: int) -> None:
    """Validate genome-scan geometry."""
    if W < w:
        raise ValueError(f"window ({W}) must be at least the motif width ({w})")
    if not 1 <= s <= W:
        raise ValueError(f"step must satisfy 1 <= step <= window, got step={s}, window={W}")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError(f"min_valid_fraction must lie in [0, 1], got {min_valid_fraction}")
    if chunk_size < s:
        raise ValueError(f"chunk_size ({chunk_size}) must be at least the step ({s})")
