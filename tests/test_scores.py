import numpy as np
import pytest

from conftest import random_counts, random_seq
from reference_impl import scores as ref_scores
from seqaffinity.encode import encode, revcomp_ints
from seqaffinity.pwm import PWM, load_pwm, to_energy
from seqaffinity.scores import max_site, partition_fn, tba, tba_per_window, trap
from seqaffinity.windows import window_energies


def _setup(rng, w=None, lam=0.7):
    w = w or int(rng.integers(5, 13))
    em = to_energy(PWM(random_counts(rng, w) + 1), lam=lam)
    return em


def _all(E, em, R0):
    return np.array(
        [
            max_site(E, em).value,
            tba(E).value,
            tba_per_window(E).value,
            trap(E, em, R0).value,
            partition_fn(E, em, R0).value,
        ]
    )


def test_match_reference_implementation(rng):
    for _ in range(25):
        w = int(rng.integers(4, 12))
        counts = random_counts(rng, w) + rng.integers(0, 2)
        lam = rng.uniform(0.5, 1.5)
        R0 = float(rng.uniform(0.01, 20))
        seq = random_seq(rng, int(rng.integers(w, 250)), p_ambig=0.02)
        em = to_energy(PWM(counts), lam=lam)
        E = window_energies(encode(seq), em)
        ref = ref_scores(seq, counts.tolist(), lam=lam, R0=R0)
        got = _all(E, em, R0)
        want = [ref["max_score"], ref["ln_tba"], ref["tba_per_window"], ref["trap"], ref["partition_fn"]]
        np.testing.assert_allclose(got, want, rtol=1e-10)


def test_revcomp_invariance(rng):
    em = _setup(rng)
    for _ in range(10):
        s = encode(random_seq(rng, 200, p_ambig=0.01))
        a = _all(window_energies(s, em), em, 2.0)
        b = _all(window_energies(revcomp_ints(s), em), em, 2.0)
        np.testing.assert_allclose(a, b, rtol=1e-12)


def test_appending_never_decreases(rng):
    em = _setup(rng)
    for _ in range(10):
        s = random_seq(rng, 150)
        t = s + random_seq(rng, 50)
        a = _all(window_energies(encode(s), em), em, 1.0)
        b = _all(window_energies(encode(t), em), em, 1.0)
        # max, ln_tba, trap, pf never decrease (tba_per_window may).
        assert np.all(b[[0, 1, 3, 4]] >= a[[0, 1, 3, 4]] - 1e-12)


def test_ordering_invariants(rng):
    for _ in range(20):
        em = _setup(rng)
        R0 = float(10 ** rng.uniform(-3, 2))
        E = window_energies(encode(random_seq(rng, 200)), em)
        ln_tba = tba(E).value
        assert -E.min() <= ln_tba + 1e-12
        t = trap(E, em, R0).value
        pf = partition_fn(E, em, R0).value
        assert pf <= t + 1e-12
        assert t <= R0 * np.exp(ln_tba) * (1 + 1e-12)


def test_low_r0_limits(rng):
    em = _setup(rng)
    E = window_energies(encode(random_seq(rng, 200)), em)
    R0 = 1e-12
    T = np.exp(tba(E).value)
    assert trap(E, em, R0).value / R0 == pytest.approx(T, rel=1e-9)
    assert partition_fn(E, em, R0).value / R0 == pytest.approx(T, rel=1e-9)


def test_lambda_preserves_window_ranking(rng):
    counts = random_counts(rng, 8) + 1
    seq = encode(random_seq(rng, 200))
    e1 = window_energies(seq, to_energy(PWM(counts), lam=0.7))
    e2 = window_energies(seq, to_energy(PWM(counts), lam=1.4))
    np.testing.assert_array_equal(np.argsort(e1.ravel(), kind="stable"), np.argsort(e2.ravel(), kind="stable"))


def test_trap_saturates_on_palindromic_sites(arnt_path, rng):
    em = to_energy(load_pwm(arnt_path), pseudocount=0.01)
    # Ten well-separated CACGTG sites in a background free of near-sites.
    spacer = "AT" * 7
    seq = spacer + (("CACGTG" + spacer) * 10)
    E = window_energies(encode(seq), em)
    assert trap(E, em, R0=1e3).value == pytest.approx(20, abs=0.5)
    # As R0 → ∞ every window saturates, not just the sites.
    assert trap(E, em, R0=1e300).value == pytest.approx(E.size, rel=1e-6)


def test_empty_sequence_scores(rng):
    em = _setup(rng, w=8)
    E = window_energies(encode("ACG"), em)
    v = _all(E, em, 1.0)
    assert np.isnan(v[0]) and v[1] == -np.inf and np.isnan(v[2]) and v[3] == 0 and v[4] == 0


def test_profiles_sum_to_value(rng):
    em = _setup(rng)
    E = window_energies(encode(random_seq(rng, 200)), em)
    for f in (trap, partition_fn):
        r = f(E, em, 3.0)
        assert r.profile.shape == E.shape
        assert r.profile.sum() == pytest.approx(r.value, rel=1e-12)
