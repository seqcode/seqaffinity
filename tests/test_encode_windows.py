import numpy as np
import pytest

from conftest import random_counts, random_seq
from reference_impl import energy_matrix
from reference_impl import window_energies as ref_windows
from seqaffinity.encode import base_composition, decode, encode, revcomp_ints
from seqaffinity.pwm import PWM, to_energy
from seqaffinity.windows import valid_windows, window_energies


def test_encode_mapping():
    np.testing.assert_array_equal(encode("ACGTacgtNnRyU"), [0, 1, 2, 3, 0, 1, 2, 3, -1, -1, -1, -1, 3])
    np.testing.assert_array_equal(encode("ACgt", hard_mask=True), [0, 1, -1, -1])
    assert encode("").shape == (0,)
    with pytest.raises(ValueError):
        encode("ACGX")


def test_revcomp_and_decode():
    s = encode("AACGTN")
    assert decode(revcomp_ints(s)) == "NACGTT"
    np.testing.assert_array_equal(base_composition(s), [2, 1, 1, 1])


def test_window_energies_match_naive_loop(rng):
    for _ in range(50):
        w = int(rng.integers(3, 12))
        counts = random_counts(rng, w) + 1
        bg = rng.dirichlet(np.ones(4) * 4)
        lam = rng.uniform(0.5, 1.5)
        seq = random_seq(rng, int(rng.integers(w, 120)), p_ambig=0.05)
        em = to_energy(PWM(counts), lam=lam, background=bg)
        E = window_energies(encode(seq), em)
        Eref, Ebg_ref, _ = energy_matrix(counts.tolist(), lam, "auto", bg.tolist())
        plus, minus = ref_windows(seq, Eref, Ebg_ref)
        np.testing.assert_allclose(E[0], plus, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(E[1], minus, rtol=1e-12, atol=1e-12)


def test_short_sequence_gives_empty(rng):
    em = to_energy(PWM(random_counts(rng, 8) + 1))
    assert window_energies(encode("ACGT"), em).shape == (2, 0)
    assert valid_windows(encode("ACGT"), 8).shape == (0,)


def test_minus_row_is_reversed_plus_of_revcomp(rng):
    em = to_energy(PWM(random_counts(rng, 7) + 1))
    for _ in range(20):
        s = encode(random_seq(rng, 90, p_ambig=0.03))
        E = window_energies(s, em)
        Erc = window_energies(revcomp_ints(s), em)
        np.testing.assert_array_equal(E[1], Erc[0][::-1])
        np.testing.assert_array_equal(E[0], Erc[1][::-1])


def test_palindromic_consensus_both_strands_zero(arnt_path):
    from seqaffinity import load_pwm

    em = to_energy(load_pwm(arnt_path))
    E = window_energies(encode("TTTCACGTGTTT"), em)
    assert E[0, 3] == 0.0 and E[1, 3] == 0.0


def test_plus_only_minus_row_infinite(rng):
    em = to_energy(PWM(random_counts(rng, 5) + 1))
    E = window_energies(encode(random_seq(rng, 30)), em, strands="plus")
    assert np.all(np.isinf(E[1])) and np.all(np.isfinite(E[0]))


def test_valid_windows():
    v = valid_windows(encode("ACGTNACGTA"), 4)
    np.testing.assert_array_equal(v, [True, False, False, False, False, True, True])
