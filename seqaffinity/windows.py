"""Per-window binding energies on both strands, in forward coordinates."""

from __future__ import annotations

import numpy as np

from .pwm import EnergyMatrix


def window_energies(seq: np.ndarray, em: EnergyMatrix, strands: str = "both") -> np.ndarray:
    """Energy of every motif window on both strands.

    Parameters
    ----------
    seq : np.ndarray
        int8 encoded sequence (0..3 = ACGT, -1 = ambiguity code).
    em : EnergyMatrix
    strands : {"both", "plus"}
        With ``"plus"`` the minus row is ``+inf`` (affinity 0).

    Returns
    -------
    np.ndarray
        ``(2, n_win)`` float64 with ``n_win = max(L − w + 1, 0)``. Row 0 is the plus
        strand; row 1 is the minus strand, stored at the forward start of the
        ``w`` bases it covers: ``E⁻[i] = Σ_j E[j, comp(s[i + w − 1 − j])]``.

    Notes
    -----
    Energies are accumulated one motif column at a time, so memory is O(L)
    regardless of motif width. Ambiguity codes take ``E_bg[j]``.
    """
    seq = np.asarray(seq)
    w = em.width
    n = seq.shape[0] - w + 1
    if n <= 0:
        return np.zeros((2, 0))
    # Map -1 → 4 so the extra column (E_bg) is selected by a plain gather.
    idx = np.where(seq < 0, 4, seq).astype(np.intp)
    out = np.zeros((2, n))
    plus, minus = out[0], out[1]
    E_ext, Ec_ext = em.E_ext, em.Ec_ext
    for j in range(w):
        plus += E_ext[j][idx[j : j + n]]
    if strands == "plus":
        minus[:] = np.inf
        return out
    for j in range(w):
        k = w - 1 - j
        minus += Ec_ext[j][idx[k : k + n]]
    return out


def valid_windows(seq: np.ndarray, w: int) -> np.ndarray:
    """Boolean ``(n_win,)``: True where the motif window has no ambiguity code."""
    seq = np.asarray(seq)
    n = seq.shape[0] - w + 1
    if n <= 0:
        return np.zeros(0, dtype=bool)
    bad = np.concatenate([[0], np.cumsum(seq < 0, dtype=np.int64)])
    return (bad[w : w + n] - bad[:n]) == 0


def log_odds_windows(E_win: np.ndarray, em: EnergyMatrix) -> np.ndarray:
    """Natural-log log-odds window scores, ``Σ_j max LO − λ·E``."""
    return em.lo_max_sum - em.lam * E_win
