import time

import numpy as np
import pytest

from conftest import random_counts, random_seq
from seqaffinity.aggregate import aggregate, segment_reduce
from seqaffinity.encode import encode
from seqaffinity.pwm import PWM, to_energy
from seqaffinity.scores import max_site, partition_fn, tba, trap
from seqaffinity.streams import streams
from seqaffinity.windows import valid_windows, window_energies


def test_stream_sums_equal_scorers(rng):
    em = to_energy(PWM(random_counts(rng, 10) + 1))
    s = encode(random_seq(rng, 10_000, p_ambig=0.001))
    E = window_energies(s, em)
    R0 = 1.7
    st = streams(E, em, np.log(R0), valid=valid_windows(s, 10))
    assert np.log(st["k"].sum()) == pytest.approx(tba(E).value, rel=1e-9)
    assert st["occ"].sum() == pytest.approx(trap(E, em, R0).value, rel=1e-9)
    assert st["pf_occ"].sum() == pytest.approx(partition_fn(E, em, R0).value, rel=1e-9)
    assert st["best"].max() == max_site(E, em).value
    assert st["valid"].sum() == valid_windows(s, 10).sum()


def test_segment_reduce_matches_naive(rng):
    x = rng.random(1000)
    starts = rng.integers(0, 900, size=200)
    n = rng.integers(1, 100, size=200)
    got = segment_reduce(x, starts, n)
    want = [x[a : a + k].sum() for a, k in zip(starts, n)]
    np.testing.assert_allclose(got, want, rtol=1e-13)
    gotm = segment_reduce(x, starts, n, np.maximum)
    np.testing.assert_array_equal(gotm, [x[a : a + k].max() for a, k in zip(starts, n)])
    # Segment ending exactly at the end of the array.
    assert segment_reduce(x, [990], [10])[0] == pytest.approx(x[990:].sum())
    with pytest.raises(ValueError):
        segment_reduce(x, [995], [10])
    with pytest.raises(ValueError):
        segment_reduce(x, [5], [0])


def test_segment_sums_avoid_cancellation():
    # Tiny values after a large prefix: prefix-sum differencing would return ~0 or noise.
    x = np.concatenate([np.full(1_000_000, 1.0), np.full(100, 1e-20)])
    got = segment_reduce(x, [1_000_000], [100])[0]
    assert got == pytest.approx(1e-18, rel=1e-12)
    prefix = np.concatenate([[0.0], np.cumsum(x)])
    naive = prefix[-1] - prefix[1_000_000]
    assert abs(naive - 1e-18) / 1e-18 > 0.5  # the failure mode we avoid (here it is exactly 0)


def test_aggregate_uses_max_for_best(rng):
    st = {"k": rng.random(100), "best": rng.normal(size=100)}
    agg = aggregate(st, np.array([0, 50]), np.array([50, 50]))
    assert agg["best"][1] == st["best"][50:].max()
    assert agg["k"][0] == pytest.approx(st["k"][:50].sum())


@pytest.mark.slow
def test_streams_20mb_under_10s(rng):
    em = to_energy(PWM(random_counts(rng, 10) + 1))
    s = rng.integers(0, 4, size=20_000_000).astype(np.int8)
    t = time.perf_counter()
    E = window_energies(s, em)
    st = streams(E, em, 0.18, want=("k", "occ", "best", "valid"), valid=valid_windows(s, 10))
    elapsed = time.perf_counter() - t
    assert len(st["k"]) == s.size - 9
    assert elapsed < 10.0, elapsed
