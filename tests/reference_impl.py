"""Deliberately naive pure-Python implementation of the five scores.

Written directly from the formulas in the README "Method" section, with plain
loops and floats and no code shared with the package, so it can serve as an
independent reference.
"""

from __future__ import annotations

import math

BASES = "ACGT"
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def energy_matrix(counts, lam=0.7, pseudocount="auto", background=(0.25, 0.25, 0.25, 0.25)):
    """Return (E, E_bg, lo_max_sum) as nested lists."""
    q = [b / sum(background) for b in background]
    E, E_bg, lo_max_sum = [], [], 0.0
    for row in counts:
        N = sum(row)
        p = math.sqrt(N) if pseudocount == "auto" else pseudocount
        lo = [math.log((row[b] + p * q[b]) / (N + p)) - math.log(q[b]) for b in range(4)]
        m = max(lo)
        lo_max_sum += m
        e = [(m - x) / lam for x in lo]
        E.append(e)
        E_bg.append(sum(q[b] * e[b] for b in range(4)))
    return E, E_bg, lo_max_sum


def _site_energy(E, E_bg, site):
    total = 0.0
    for j, c in enumerate(site):
        total += E[j][BASES.index(c)] if c in BASES else E_bg[j]
    return total


def window_energies(seq, E, E_bg):
    """(plus, minus) lists; minus at the forward start of the bases it covers."""
    seq = seq.upper().replace("U", "T")
    w = len(E)
    plus, minus = [], []
    for i in range(len(seq) - w + 1):
        site = seq[i : i + w]
        rc = "".join(COMP.get(c, "N") for c in reversed(site))
        plus.append(_site_energy(E, E_bg, site))
        minus.append(_site_energy(E, E_bg, rc))
    return plus, minus


def scores(seq, counts, lam=0.7, pseudocount="auto", background=(0.25,) * 4, R0=None):
    """Dict with max_score, ln_tba, tba_per_window, trap, trap_per_bp, partition_fn."""
    E, E_bg, lo_max_sum = energy_matrix(counts, lam, pseudocount, background)
    w = len(E)
    if R0 is None:
        R0 = min(math.exp(0.584 * w - 5.66), 10.0)
    plus, minus = window_energies(seq, E, E_bg)
    n = len(plus)
    L = len(seq)
    allE = plus + minus
    K = [math.exp(-e) for e in allE]
    tba = sum(K)
    out = {
        "max_score": lo_max_sum - lam * min(allE) if n else float("nan"),
        "ln_tba": math.log(tba) if tba > 0 else float("-inf"),
        "tba_per_window": tba / (2 * n) if n else float("nan"),
        "trap": sum(R0 * k / (1 + R0 * k) for k in K),
    }
    out["trap_per_bp"] = out["trap"] / L if L else float("nan")
    # Partition function in plain floats (fine for short sequences).
    a = [R0 * (math.exp(-plus[i]) + math.exp(-minus[i])) for i in range(n)]
    zf = [1.0] * (L + 1)
    for x in range(w, L + 1):
        zf[x] = zf[x - 1] + a[x - w] * zf[x - w]
    zb = [1.0] * (L + 2)
    for y in range(n - 1, -1, -1):
        zb[y] = zb[y + 1] + a[y] * zb[y + w]
    out["partition_fn"] = sum(a[i] * zf[i] * zb[i + w] / zf[L] for i in range(n))
    return out
