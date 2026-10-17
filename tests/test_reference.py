"""Cross-checks against published implementations.

The reference numbers come from ``scripts/make_reference_fixtures.R`` (a manual
step that needs R/Bioconductor). Each test skips when its fixture is missing.

Neither tool uses seqaffinity's default conventions, so each test rebuilds the
tool's convention from seqaffinity's window energies and compares at 1e-9. That
checks the energy matrix, both strands' window energies and their forward-start
coordinates. The differences (read from the tools' C sources) are:

MatrixRider ``getSeqOccupancy(seq, pfm, cutoff=0)``
    * pseudocount: zero counts are *replaced* by 1, non-zero counts are left
      alone, no other pseudocount. seqaffinity's additive rule can't express
      this, so the test passes the replaced counts with ``pseudocount=0``.
    * strands: each window contributes ``max(exp(LO+), exp(LO-))``, the better
      strand only. seqaffinity's TBA sums both strands.
    * ambiguity: a window containing N contributes 0. seqaffinity scores N at
      the background-mean energy.
    * λ = 1 and a uniform background (``PFMatrix`` default), as expected.
    With seqaffinity's defaults (``auto`` pseudocount, summed strands), the
    rescaled TBA ``exp(ln_tba + lo_max_sum)`` is about 1.77× MatrixRider's
    value on ten.fa and does not keep its rank order (seq7/seq8 swap).

tRap ``affinity(pwm, seq)``
    * pseudocount +1 per cell, λ = 0.7, gc.content = 0.5, Rmax from the width
      formula: identical to seqaffinity with ``pseudocount=4``, a uniform background
      and the default R0.
    * strands: per window ``P+ + (1 − P+)·P-`` (at least one strand bound).
      seqaffinity sums ``P+ + P-``. The difference is ``Σ P+·P-``, about 5% on
      ten.fa because Arnt (CACGTG) is palindromic.
    * ambiguity: N is scored as A, and only a window whose last base is N is
      zeroed (``illegalBase`` is reset per column in trap.cpp). The R script
      therefore reports sums over N-free windows (``*_valid``).
"""

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES
from seqaffinity import PWM, load_pwm, score_one, score_sequences, to_energy
from seqaffinity.encode import encode
from seqaffinity.genome import read_fasta


def _load(name):
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} not generated (run scripts/make_reference_fixtures.R)")
    return pd.read_csv(path).set_index("seq_id")


def _records():
    return read_fasta(FIXTURES / "ten.fa")


def _valid_windows(seq: str, w: int) -> np.ndarray:
    """True for motif windows that contain no ambiguity code."""
    bad = (encode(seq) < 0).astype(np.int64)
    return np.convolve(bad, np.ones(w, dtype=np.int64), mode="valid") == 0


def test_matrixrider_total_affinity():
    ref = _load("matrixrider.csv")
    pwm = load_pwm(FIXTURES / "MA0004.1.jaspar")
    counts = np.where(pwm.counts == 0, 1.0, pwm.counts)  # MatrixRider's zero → 1
    em = to_energy(
        PWM(counts=counts, name=pwm.name, is_probability=False, depth=None),
        lam=1.0,
        pseudocount=0,
    )
    for seq_id, seq in _records():
        E_win = score_one(em, seq)["E_win"]
        best_strand = np.exp(em.lo_max_sum - E_win).max(axis=0)
        ours = best_strand[_valid_windows(seq, em.E.shape[0])].sum()
        np.testing.assert_allclose(ours, ref.loc[seq_id, "total_affinity"], rtol=1e-9)


def test_trap_affinity():
    """seqaffinity's own TRAP score (strands summed) equals tRap per strand."""
    ref = _load("trap.csv")
    em = to_energy(load_pwm(FIXTURES / "MA0004.1.jaspar"), lam=0.7, pseudocount=4)
    df = score_sequences(em, FIXTURES / "ten.fa", scores=("trap",)).set_index("seq_id")
    for seq_id, seq in _records():
        ok = _valid_windows(seq, em.E.shape[0])
        expected = ref.loc[seq_id, "trap_sum_valid"]
        if ok.all():
            np.testing.assert_allclose(df.loc[seq_id, "trap"], expected, rtol=1e-9)
        profile = score_one(em, seq)["trap"].profile
        np.testing.assert_allclose(profile[:, ok].sum(), expected, rtol=1e-9)


def test_trap_native_strand_combination():
    """tRap's default output, rebuilt from seqaffinity's per-strand occupancies."""
    ref = _load("trap.csv")
    em = to_energy(load_pwm(FIXTURES / "MA0004.1.jaspar"), lam=0.7, pseudocount=4)
    for seq_id, seq in _records():
        ok = _valid_windows(seq, em.E.shape[0])
        p_plus, p_minus = score_one(em, seq)["trap"].profile
        either = p_plus + (1.0 - p_plus) * p_minus
        np.testing.assert_allclose(either[ok].sum(), ref.loc[seq_id, "trap_or_valid"], rtol=1e-9)
        if ok.all():
            np.testing.assert_allclose(
                ref.loc[seq_id, "trap_native"], ref.loc[seq_id, "trap_or_valid"], rtol=1e-12
            )
