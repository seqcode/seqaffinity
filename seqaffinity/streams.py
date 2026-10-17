"""Per-position streams from which every window score is a segment sum or maximum."""

from __future__ import annotations

import numpy as np

from .pf import ln_activity, pf_occupancy_1d
from .pwm import EnergyMatrix
from .scores import trap_profile
from .windows import log_odds_windows

STREAM_KEYS = ("k", "occ", "best", "pf_occ", "valid")


def streams(
    E_win: np.ndarray,
    em: EnergyMatrix,
    ln_r0: float,
    want=STREAM_KEYS,
    valid: np.ndarray | None = None,
    pf_backend: str = "auto",
) -> dict[str, np.ndarray]:
    """Per-motif-window streams, all of length ``n_win``.

    Parameters
    ----------
    E_win : np.ndarray
        ``(2, n_win)`` energies in forward coordinates.
    em : EnergyMatrix
    ln_r0 : float
        ``ln R0``.
    want : iterable of str
        Subset of ``k`` (Σ strands K), ``occ`` (Σ strands TRAP occupancy),
        ``best`` (max-strand log-odds), ``pf_occ`` (partition-function occupancy
        over the whole array, both strands), ``valid``.
    valid : np.ndarray or None
        Boolean validity per window; required if ``"valid"`` is wanted.
    pf_backend : str
        Passed to :func:`seqaffinity.pf.pf_occupancy_1d`.

    Returns
    -------
    dict[str, np.ndarray]
    """
    want = set(want)
    out: dict[str, np.ndarray] = {}
    if "k" in want:
        K = np.exp(-E_win)
        out["k"] = K[0] + K[1]
    if "occ" in want:
        o = trap_profile(E_win, ln_r0)
        out["occ"] = o[0] + o[1]
    if "best" in want:
        out["best"] = log_odds_windows(np.minimum(E_win[0], E_win[1]), em)
    if "pf_occ" in want:
        out["pf_occ"] = pf_occupancy_1d(ln_activity(E_win, ln_r0), em.width, backend=pf_backend)[0]
    if "valid" in want:
        if valid is None:
            raise ValueError("valid mask required for the 'valid' stream")
        out["valid"] = valid.astype(np.int64)
    return out
