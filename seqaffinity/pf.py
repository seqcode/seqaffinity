"""Partition function with steric exclusion (1-D hard-core lattice gas).

Let ``a_i = R0·(K⁺_i + K⁻_i)`` for motif start ``i``. With prefix-length indexing,

    Z_f[x] = Z_f[x−1] + a_{x−w}·Z_f[x−w]   (x ≥ w),   Z_f[x] = 1   (x < w)
    Z_b[y] = Z_b[y+1] + a_y·Z_b[y+w]       (y ≤ L−w), Z_b[y] = 1   (y > L−w)

and the occupancy of start ``i`` (both strands together) is
``a_i·Z_f[i]·Z_b[i+w] / Z_f[L]``. Everything is in log space.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

try:  # optional dependency
    import numba

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    numba = None
    HAVE_NUMBA = False

SLOW_FALLBACK_WARN = 1_000_000


def ln_activity(E_win: np.ndarray, ln_r0: float) -> np.ndarray:
    """``ln a_i = ln R0 + ln(exp(−E⁺_i) + exp(−E⁻_i))`` for a ``(2, n)`` energy array."""
    return ln_r0 + np.logaddexp(-E_win[0], -E_win[1])


def pf_occupancy_batch(ln_a: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Occupancies for a batch of sequences, vectorised across the batch.

    Parameters
    ----------
    ln_a : np.ndarray
        ``(B, n)`` log activities. Sequences shorter than ``n`` windows are padded
        on the right with ``-inf`` (no site), which leaves their Z unchanged.
    w : int
        Motif width.

    Returns
    -------
    occ : np.ndarray
        ``(B, n)`` per-start occupancy (both strands combined).
    ln_z : np.ndarray
        ``(B,)`` log partition function.
    """
    ln_a = np.atleast_2d(np.asarray(ln_a, dtype=np.float64))
    B, n = ln_a.shape
    if n == 0:
        return np.zeros((B, 0)), np.zeros(B)
    L = n + w - 1
    lzf = np.zeros((L + 1, B))
    lzb = np.zeros((L + 1, B))
    a_t = np.ascontiguousarray(ln_a.T)
    for x in range(w, L + 1):
        np.logaddexp(lzf[x - 1], a_t[x - w] + lzf[x - w], out=lzf[x])
    for y in range(n - 1, -1, -1):
        np.logaddexp(lzb[y + 1], a_t[y] + lzb[y + w], out=lzb[y])
    ln_z = lzf[L]
    occ = np.exp(a_t + lzf[:n] + lzb[w : w + n] - ln_z[None, :])
    return np.ascontiguousarray(occ.T), ln_z.copy()


def _pf_occupancy_1d_py(ln_a: np.ndarray, w: int):
    occ, ln_z = pf_occupancy_batch(ln_a[None, :], w)
    return occ[0], float(ln_z[0])


if HAVE_NUMBA:

    @numba.njit(cache=True)
    def _lae(a, b):  # scalar logaddexp
        if a == -np.inf:
            return b
        if b == -np.inf:
            return a
        if a > b:
            return a + math.log1p(math.exp(b - a))
        return b + math.log1p(math.exp(a - b))

    @numba.njit(cache=True)
    def _pf_occupancy_1d_nb(ln_a, w):
        n = ln_a.shape[0]
        L = n + w - 1
        lzf = np.zeros(L + 1)
        lzb = np.zeros(L + 1)
        for x in range(w, L + 1):
            lzf[x] = _lae(lzf[x - 1], ln_a[x - w] + lzf[x - w])
        for y in range(n - 1, -1, -1):
            lzb[y] = _lae(lzb[y + 1], ln_a[y] + lzb[y + w])
        ln_z = lzf[L]
        occ = np.empty(n)
        for i in range(n):
            occ[i] = math.exp(ln_a[i] + lzf[i] + lzb[i + w] - ln_z)
        return occ, ln_z


def pf_occupancy_1d(ln_a: np.ndarray, w: int, backend: str = "auto") -> tuple[np.ndarray, float]:
    """Occupancies for one (possibly very long) sequence.

    Parameters
    ----------
    ln_a : np.ndarray
        ``(n,)`` log activities.
    w : int
        Motif width.
    backend : {"auto", "numba", "numpy"}
        ``"auto"`` uses numba when installed. The NumPy path loops in Python and
        warns above 1 Mb.

    Returns
    -------
    occ : np.ndarray
        ``(n,)`` per-start occupancy.
    ln_z : float
    """
    ln_a = np.ascontiguousarray(ln_a, dtype=np.float64)
    if ln_a.shape[0] == 0:
        return np.zeros(0), 0.0
    if backend == "auto":
        backend = "numba" if HAVE_NUMBA else "numpy"
    if backend == "numba":
        if not HAVE_NUMBA:
            raise RuntimeError("numba is not installed")
        occ, ln_z = _pf_occupancy_1d_nb(ln_a, int(w))
        return occ, float(ln_z)
    if ln_a.shape[0] > SLOW_FALLBACK_WARN:
        warnings.warn(
            f"partition function on {ln_a.shape[0]:,} positions without numba is slow; "
            "install numba (pip install 'seqaffinity[fast]')",
            stacklevel=2,
        )
    return _pf_occupancy_1d_py(ln_a, w)
