"""The five whole-sequence scores, each computed from a ``(2, n_win)`` energy array."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import trap_r0
from .pf import ln_activity, pf_occupancy_1d
from .pwm import EnergyMatrix
from .windows import log_odds_windows


@dataclass(frozen=True)
class ScoreResult:
    """A score value with an optional per-window profile.

    Attributes
    ----------
    value : float
    profile : np.ndarray or None
        ``(2, n_win)`` per-window occupancy (plus, minus) for TRAP and PF.
    units : str
    """

    value: float
    profile: np.ndarray | None = None
    units: str = ""


def _n_strands(E_win: np.ndarray) -> int:
    return 1 if E_win.shape[1] and np.all(np.isinf(E_win[1])) else 2


def resolve_r0(R0: float | None, em: EnergyMatrix, warn: bool = True) -> float:
    """Return ``R0`` or TRAP's default for this motif width."""
    return trap_r0(em.width, warn=warn) if R0 is None else float(R0)


def max_site(E_win: np.ndarray, em: EnergyMatrix) -> ScoreResult:
    """Best log-odds window score over both strands (natural-log units).

    Returns NaN for a sequence shorter than the motif.
    """
    if E_win.shape[1] == 0:
        return ScoreResult(float("nan"), units="log-odds (nat)")
    return ScoreResult(float(log_odds_windows(E_win.min(), em)), units="log-odds (nat)")


def tba(E_win: np.ndarray) -> ScoreResult:
    """``ln Σ K`` over both strands (log total binding affinity).

    ``K ≤ 1`` so the linear sum cannot overflow. Returns ``-inf`` for an empty
    window set.
    """
    total = float(np.exp(-E_win).sum())
    with np.errstate(divide="ignore"):
        return ScoreResult(float(np.log(total)), units="ln(sum K)")


def tba_per_window(E_win: np.ndarray) -> ScoreResult:
    """Mean affinity per motif window: ``TBA / (n_strands·n_win)``."""
    n = E_win.shape[1] * _n_strands(E_win)
    if n == 0:
        return ScoreResult(float("nan"), units="K per window")
    return ScoreResult(float(np.exp(-E_win).sum() / n), units="K per window")


def trap(E_win: np.ndarray, em: EnergyMatrix, R0: float | None = None) -> ScoreResult:
    """TRAP expected occupancy ``Σ R0·K / (1 + R0·K)`` with per-window profile."""
    r0 = resolve_r0(R0, em)
    prof = trap_profile(E_win, np.log(r0))
    return ScoreResult(float(prof.sum()), profile=prof, units="expected bound")


def trap_profile(E_win: np.ndarray, ln_r0: float) -> np.ndarray:
    """Per-window TRAP occupancy ``expit(ln R0 − E)`` (numerically stable)."""
    x = ln_r0 - E_win
    # expit(x) = 1 / (1 + exp(-x)); E may be +inf (plus-only) → 0.
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def partition_fn(
    E_win: np.ndarray, em: EnergyMatrix, R0: float | None = None, backend: str = "auto"
) -> ScoreResult:
    """Expected number of bound molecules with steric exclusion.

    Both strands at the same start are alternative placements of one footprint.
    The profile splits each start's occupancy between strands in proportion to K.
    """
    r0 = resolve_r0(R0, em)
    if E_win.shape[1] == 0:
        return ScoreResult(0.0, profile=np.zeros((2, 0)), units="expected bound")
    ln_a = ln_activity(E_win, np.log(r0))
    occ, _ = pf_occupancy_1d(ln_a, em.width, backend=backend)
    return ScoreResult(float(occ.sum()), profile=split_strands(occ, E_win), units="expected bound")


def split_strands(occ: np.ndarray, E_win: np.ndarray) -> np.ndarray:
    """Split per-start occupancy into ``(2, n)`` by the strands' relative affinity."""
    ln_k = np.logaddexp(-E_win[0], -E_win[1])
    return occ[None, :] * np.exp(-E_win - ln_k[None, :])
