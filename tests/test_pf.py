import itertools

import numpy as np
import pytest

from conftest import random_counts, random_seq
from seqaffinity import pf
from seqaffinity.encode import encode
from seqaffinity.pwm import PWM, to_energy
from seqaffinity.scores import partition_fn, trap
from seqaffinity.windows import window_energies


def brute_force(K, R0, w):
    """Enumerate every set of non-overlapping (start, strand) placements."""
    n = K.shape[1]
    items = [(i, s) for i in range(n) for s in range(2)]
    Z = 0.0
    occ = np.zeros_like(K)
    for r in range(len(items) + 1):
        for combo in itertools.combinations(items, r):
            starts = sorted(i for i, _ in combo)
            if any(b - a < w for a, b in zip(starts, starts[1:])):
                continue
            weight = R0 ** len(combo)
            for i, s in combo:
                weight *= K[s, i]
            Z += weight
            for i, s in combo:
                occ[s, i] += weight
    return Z, occ / Z


@pytest.mark.parametrize("L", [3, 5, 8, 10, 12])
def test_brute_force(rng, L):
    w = 3
    for _ in range(3):
        em = to_energy(PWM(random_counts(rng, w, depth=10) + 1), lam=rng.uniform(0.3, 1.5))
        E = window_energies(encode(random_seq(rng, L)), em)
        R0 = float(10 ** rng.uniform(-1, 2))
        Z, occ = brute_force(np.exp(-E), R0, w)
        res = partition_fn(E, em, R0)
        np.testing.assert_allclose(res.profile, occ, rtol=1e-9, atol=1e-15)
        ln_a = pf.ln_activity(E, np.log(R0))
        _, ln_z = pf.pf_occupancy_batch(ln_a[None], w)
        assert ln_z[0] == pytest.approx(np.log(Z), rel=1e-9, abs=1e-12)


@pytest.mark.skipif(not pf.HAVE_NUMBA, reason="numba not installed")
def test_numba_matches_numpy(rng):
    ln_a = rng.normal(-3, 3, size=20_000)
    o1, z1 = pf.pf_occupancy_1d(ln_a, 9, backend="numba")
    o2, z2 = pf.pf_occupancy_1d(ln_a, 9, backend="numpy")
    np.testing.assert_allclose(o1, o2, rtol=1e-9, atol=1e-300)
    assert z1 == pytest.approx(z2, rel=1e-12)


def test_batch_padding_does_not_change_results(rng):
    w = 6
    rows = [rng.normal(-2, 2, size=n) for n in (50, 80, 120)]
    m = 120
    batch = np.full((3, m), -np.inf)
    for r, x in enumerate(rows):
        batch[r, : len(x)] = x
    occ, z = pf.pf_occupancy_batch(batch, w)
    for r, x in enumerate(rows):
        o1, z1 = pf.pf_occupancy_batch(x[None], w)
        np.testing.assert_array_equal(occ[r, : len(x)], o1[0])
        assert np.all(occ[r, len(x) :] == 0)
        assert z[r] == z1[0]


def test_two_overlapping_sites():
    # Toy motif AAAAA; two consensus sites overlapping by one base.
    counts = np.tile([100.0, 0, 0, 0], (5, 1))
    em = to_energy(PWM(counts), pseudocount=0.01)
    seq = "CCCCC" + "AAAAAAAAA" + "CCCCC"  # sites at 5 and 9 overlap by one base
    E = window_energies(encode(seq), em)
    R0 = 5.0
    single = R0 / (1 + R0)
    p = partition_fn(E, em, R0).value
    t = trap(E, em, R0).value
    assert single < p < t


def test_high_r0_saturates_at_packing_limit(rng):
    em = to_energy(PWM(random_counts(rng, 8) + 1))
    L = 400
    E = window_energies(encode(random_seq(rng, L)), em)
    v = partition_fn(E, em, 1e30).value
    assert L / 8 - 1 <= v <= L / 8 + 1e-9


def test_fallback_warns_for_long_inputs(monkeypatch):
    monkeypatch.setattr(pf, "SLOW_FALLBACK_WARN", 10)
    with pytest.warns(UserWarning):
        pf.pf_occupancy_1d(np.zeros(20), 3, backend="numpy")
