import time

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES, random_seq
from seqaffinity import load_pwm, score_one, score_sequences
from seqaffinity.cli import main

COLS = ["max_score", "ln_tba", "tba_per_window", "trap", "trap_per_bp", "partition_fn"]


def _read_tsv(path_or_text):
    return pd.read_csv(path_or_text, sep="\t", comment="#")


def test_score_sequences_matches_expected():
    df = score_sequences(FIXTURES / "MA0004.1.jaspar", FIXTURES / "ten.fa")
    exp = _read_tsv(FIXTURES / "expected.tsv")
    assert list(df["seq_id"]) == list(exp["seq_id"])
    np.testing.assert_allclose(df[COLS].to_numpy(), exp[COLS].to_numpy(), rtol=1e-9)
    assert df.attrs["lambda"] == 0.7 and df.attrs["width"] == 6


def test_cli_score_reproduces_expected(tmp_path, capsys):
    out = tmp_path / "out.tsv"
    assert main(["score", str(FIXTURES / "MA0004.1.jaspar"), str(FIXTURES / "ten.fa"), "--out", str(out)]) == 0
    got = _read_tsv(out)
    exp = _read_tsv(FIXTURES / "expected.tsv")
    np.testing.assert_allclose(got[COLS].to_numpy(), exp[COLS].to_numpy(), rtol=1e-9)
    header = [l for l in out.read_text().splitlines() if l.startswith("#")]
    assert any("R0=" in l for l in header) and any("background=" in l for l in header)


def test_cli_profile(tmp_path):
    prof = tmp_path / "p.tsv"
    main(
        [
            "score",
            str(FIXTURES / "MA0004.1.jaspar"),
            str(FIXTURES / "ten.fa"),
            "--out",
            str(tmp_path / "o.tsv"),
            "--profile",
            "seq3",
            "--profile-out",
            str(prof),
        ]
    )
    p = pd.read_csv(prof, sep="\t")
    assert set(p["strand"]) == {"+", "-"}
    exp = _read_tsv(FIXTURES / "expected.tsv").set_index("seq_id").loc["seq3"]
    assert p["trap_occ"].sum() == pytest.approx(exp["trap"], rel=1e-9)
    assert p["pf_occ"].sum() == pytest.approx(exp["partition_fn"], rel=1e-9)


def test_inputs_and_subsets(arnt_path, rng):
    seqs = [random_seq(rng, 150) for _ in range(5)]
    a = score_sequences(arnt_path, seqs)
    b = score_sequences(arnt_path, {f"s{i}": s for i, s in enumerate(seqs)})
    np.testing.assert_array_equal(a[COLS].to_numpy(), b[COLS].to_numpy())
    c = score_sequences(arnt_path, seqs, scores=("max", "pf"))
    assert [k for k in c.columns if k in COLS] == ["max_score", "partition_fn"]
    np.testing.assert_array_equal(c["partition_fn"], a["partition_fn"])
    with pytest.raises(ValueError):
        score_sequences(arnt_path, seqs, scores=("bogus",))


def test_mixed_lengths_and_profiles(arnt_path, rng):
    seqs = [random_seq(rng, int(n)) for n in rng.integers(3, 600, size=40)]
    df, prof = score_sequences(arnt_path, seqs, return_profiles=True)
    for i, s in enumerate(seqs):
        one = score_one(arnt_path, s)
        assert df["partition_fn"][i] == pytest.approx(one["pf"].value, rel=1e-10, abs=1e-300)
        assert df["trap"][i] == pytest.approx(one["trap"].value, rel=1e-12, abs=1e-300)
        np.testing.assert_allclose(prof[f"seq{i + 1}"]["pf"], one["pf"].profile, rtol=1e-10, atol=1e-300)


def test_background_options(arnt_path, rng):
    seqs = [random_seq(rng, 200) for _ in range(3)]
    u = score_sequences(arnt_path, seqs)
    f = score_sequences(arnt_path, seqs, background="from-seqs")
    e = score_sequences(arnt_path, seqs, background="0.3,0.2,0.2,0.3")
    assert not np.allclose(u["ln_tba"], e["ln_tba"])
    assert f.attrs["background"] != u.attrs["background"]


def test_plus_strand_only(arnt_path, rng):
    seqs = [random_seq(rng, 200) for _ in range(3)]
    both = score_sequences(arnt_path, seqs)
    plus = score_sequences(arnt_path, seqs, strands="plus")
    assert np.all(plus["ln_tba"] <= both["ln_tba"])
    assert np.all(plus["partition_fn"] <= both["partition_fn"] + 1e-12)


@pytest.mark.slow
def test_performance_10k_sequences(arnt_path, rng):
    seqs = ["".join(rng.choice(list("ACGT"), size=200)) for _ in range(10_000)]
    pwm = load_pwm(arnt_path)
    score_sequences(pwm, seqs[:10])  # warm-up
    t = time.perf_counter()
    score_sequences(pwm, seqs)
    assert time.perf_counter() - t < 5.0
