import math
import warnings

import numpy as np
import pytest

from conftest import FIXTURES, random_counts
from seqaffinity.params import trap_r0
from seqaffinity.pwm import PWM, load_pwm, parse_counts, parse_jaspar, parse_meme, to_energy

ARNT = np.array(
    [[4, 16, 0, 0], [19, 0, 1, 0], [0, 20, 0, 0], [0, 0, 20, 0], [0, 0, 0, 20], [0, 0, 20, 0]], dtype=float
)


def test_jaspar_parser():
    p = load_pwm(FIXTURES / "MA0004.1.jaspar")
    assert p.name == "MA0004.1"
    assert not p.is_probability
    np.testing.assert_array_equal(p.counts, ARNT)


def test_meme_parser_uses_nsites_as_depth():
    p = load_pwm(FIXTURES / "MA0004.1.meme")
    assert p.is_probability and p.depth == 20
    np.testing.assert_allclose(p.counts, ARNT, atol=1e-9)


def test_counts_parser():
    p = load_pwm(FIXTURES / "MA0004.1.counts.txt")
    np.testing.assert_array_equal(p.counts, ARNT)


def test_bare_jaspar_rows_and_transpose():
    text = "\n".join(" ".join(str(int(x)) for x in row) for row in ARNT.T)
    np.testing.assert_array_equal(parse_jaspar(text).counts, ARNT)
    np.testing.assert_array_equal(parse_counts(text).counts, ARNT)


def test_select_motif_by_name():
    p = load_pwm(FIXTURES / "two_motifs.jaspar", name="Toy")
    assert p.width == 3 and p.name == "TEST.1"
    with pytest.raises(ValueError):
        load_pwm(FIXTURES / "two_motifs.jaspar", name="nope")


def test_probability_matrix_default_depth():
    probs = ARNT / 20
    p = PWM.from_matrix(probs)
    assert p.is_probability and p.depth == 100
    np.testing.assert_allclose(p.counts, probs * 100)


@pytest.mark.parametrize(
    "bad",
    [
        "MEME version 4\nALPHABET= ACGU\nMOTIF x\nletter-probability matrix: alength= 4 w= 1\n0.25 0.25 0.25 0.25\n",
    ],
)
def test_invalid_meme_alphabet(bad):
    with pytest.raises(ValueError):
        parse_meme(bad)


def test_invalid_jaspar_alphabet_and_values():
    with pytest.raises(ValueError):
        parse_jaspar(">x\nA [1 2]\nC [1 2]\nG [1 2]\nX [1 2]\n")
    with pytest.raises(ValueError):
        PWM(np.array([[1, -1, 0, 0]]))
    with pytest.raises(ValueError):
        PWM(np.zeros((2, 3)))


def test_one_hot_energies():
    N, p, lam = 50.0, 2.0, 0.7
    counts = np.zeros((5, 4))
    cons = [0, 3, 2, 1, 0]
    counts[np.arange(5), cons] = N
    em = to_energy(PWM(counts), lam=lam, pseudocount=p)
    q = 0.25
    expected = math.log((N + p * q) / (p * q)) / lam
    for j, b in enumerate(cons):
        assert em.E[j, b] == 0.0
        others = np.delete(em.E[j], b)
        np.testing.assert_allclose(others, expected, rtol=1e-12)


def test_consensus_exactly_zero_random(rng):
    for _ in range(20):
        c = random_counts(rng, rng.integers(4, 20)) + rng.integers(0, 3)
        bg = rng.dirichlet(np.ones(4) * 5)
        em = to_energy(PWM(c), lam=rng.uniform(0.3, 2), background=bg)
        assert np.all(em.E >= 0)
        assert np.all(em.E.min(axis=1) == 0.0)
        np.testing.assert_allclose(em.E_bg, em.E @ bg)


def test_auto_pseudocount_is_sqrt_n():
    em = to_energy(PWM(ARNT))
    np.testing.assert_allclose(em.pseudocount, math.sqrt(20))


def test_trap_pseudocount_convention():
    # p = 4 with uniform background is TRAP's +1 per cell: E = ln((n_max+1)/(n_b+1))/λ.
    em = to_energy(PWM(ARNT), lam=0.7, pseudocount=4)
    n = ARNT
    ref = np.log((n.max(axis=1, keepdims=True) + 1) / (n + 1)) / 0.7
    np.testing.assert_allclose(em.E, ref, atol=1e-12)


def test_lambda_validation():
    with pytest.raises(ValueError):
        to_energy(PWM(ARNT), lam=0)


def test_trap_r0():
    assert trap_r0(10) == pytest.approx(math.exp(0.18))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trap_r0(13)
    with pytest.warns(UserWarning):
        assert trap_r0(14) == 10.0
