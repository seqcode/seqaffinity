import io

import numpy as np
import pandas as pd
import pytest

from conftest import write_fasta
from seqaffinity import load_pwm, scan_genome, score_sequences, to_energy
from seqaffinity.encode import decode, encode, revcomp_ints
from seqaffinity.pf import ln_activity, pf_occupancy_1d
from seqaffinity.windows import window_energies

COLS = ["max_score", "ln_tba", "tba_per_window", "trap", "trap_per_bp", "partition_fn"]


def run_scan(pwm, fasta, **kw):
    buf = io.StringIO()
    kw.setdefault("background", "uniform")
    scan_genome(pwm, fasta, out=buf, **kw)
    buf.seek(0)
    return pd.read_csv(buf, sep="\t", comment="#", na_values="NA", keep_default_na=False)


@pytest.fixture(scope="module")
def arnt():
    from conftest import FIXTURES

    return load_pwm(FIXTURES / "MA0004.1.jaspar")


@pytest.fixture(scope="module")
def isolated(arnt, genome_fasta):
    return run_scan(arnt, genome_fasta["gz"], W=200, s=50, pf_mode="isolated", chunk_size=1_000_000)


def test_scan_equals_score_on_extracted_windows(arnt, isolated, genome_records):
    seqs = dict(genome_records)
    df = isolated
    win = [(c, a, b) for c, a, b in zip(df["chrom"], df["start"], df["end"])]
    extracted = [(f"{c}:{a}", seqs[c][a:b]) for c, a, b in win]
    ref = score_sequences(arnt, extracted)
    na = df["max_score"].isna().to_numpy()
    assert na.sum() > 0 and (~na).sum() > 30_000
    got = df.loc[~na, COLS].to_numpy()
    want = ref.loc[~na, COLS].to_numpy()
    # Written with %.10g, so compare at that precision; exactness is checked in memory below.
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=0)


def test_scan_equals_score_in_memory(arnt, genome_records):
    """Same check without text round-tripping, at tight tolerance."""
    from seqaffinity.genome import Chunk
    from seqaffinity.scan import process_chunk

    em = to_energy(arnt)
    name, seq = genome_records[0]
    sub = seq[:200_000]
    s = encode(sub)
    starts = np.arange(0, 200_000 - 200 + 1, 50)
    ch = Chunk(name, 0, s, starts, np.full(starts.size, 200))
    _, _, _, n_valid, vals = process_chunk(ch, em, np.log(1.2), ("max", "tba", "tba_per_window", "trap", "pf"), "both", "isolated", 0.0)
    ref = score_sequences(em, [sub[a : a + 200] for a in starts], R0=1.2)
    np.testing.assert_array_equal(vals["max_score"], ref["max_score"])
    for c in ("ln_tba", "tba_per_window", "trap", "trap_per_bp"):
        np.testing.assert_allclose(vals[c], ref[c], rtol=1e-12)
    np.testing.assert_allclose(vals["partition_fn"], ref["partition_fn"], rtol=1e-9)


@pytest.mark.parametrize("cs", [100_000, 20_000_000])
def test_chunk_size_invariance(arnt, genome_fasta, isolated, cs):
    df = run_scan(arnt, genome_fasta["gz"], W=200, s=50, pf_mode="isolated", chunk_size=cs)
    pd.testing.assert_frame_equal(df, isolated, check_exact=True)


def test_worker_invariance(arnt, genome_fasta, isolated):
    df = run_scan(arnt, genome_fasta["gz"], W=200, s=50, pf_mode="isolated", chunk_size=300_000, workers=4)
    pd.testing.assert_frame_equal(df, isolated, check_exact=True)


def test_plain_and_gzip_inputs_agree(arnt, genome_fasta):
    a = run_scan(arnt, genome_fasta["plain"], W=300, s=300, scores="max,tba")
    b = run_scan(arnt, genome_fasta["gz"], W=300, s=300, scores="max,tba")
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_genome_pf_matches_whole_chromosome(arnt, genome_fasta, genome_records):
    em = to_energy(arnt)
    r0 = 1.2
    df = run_scan(arnt, genome_fasta["gz"], W=200, s=50, scores="pf", R0=r0, chunk_size=100_000, min_valid_fraction=0)
    for name, seq in genome_records[:2]:
        E = window_energies(encode(seq), em)
        occ, _ = pf_occupancy_1d(ln_activity(E, np.log(r0)), em.width)
        sub = df[df["chrom"] == name]
        want = [occ[a : a + 195].sum() for a in sub["start"]]
        np.testing.assert_allclose(sub["partition_fn"], want, rtol=1e-8, atol=1e-12)


def test_pf_overlap_doubling_changes_nothing(arnt, genome_fasta):
    a = run_scan(arnt, genome_fasta["gz"], W=200, s=50, scores="pf", chunk_size=100_000, region="chr2")
    b = run_scan(arnt, genome_fasta["gz"], W=200, s=50, scores="pf", chunk_size=100_000, region="chr2", pf_overlap=2 * 64 * 6)
    np.testing.assert_allclose(a["partition_fn"], b["partition_fn"], rtol=1e-8, atol=1e-12)


def test_genome_vs_isolated_pf_differ_only_slightly(arnt, genome_fasta, isolated):
    g = run_scan(arnt, genome_fasta["gz"], W=200, s=50, scores="pf", region="chr2")
    iso = isolated[isolated["chrom"] == "chr2"].reset_index(drop=True)
    ok = g["partition_fn"].notna()
    rel = np.abs(g["partition_fn"][ok] - iso["partition_fn"][ok]) / iso["partition_fn"][ok].clip(lower=1e-12)
    assert np.median(rel) < 1e-3
    assert np.all(g["partition_fn"][ok] <= iso["partition_fn"][ok] * 1.5 + 1e-9)


def test_n_run_na_rule(arnt, genome_fasta):
    # chr1 has an N run at [500020, 500170): window [500000, 500200) has 195 motif windows,
    # of which only those starting in [500000, 500015) and [500170, 500195) are N-free → 40.
    kw = dict(W=200, s=50, scores="tba", region="chr1:499,001-502,000")
    hi = run_scan(arnt, genome_fasta["gz"], min_valid_fraction=0.5, **kw).set_index("start")
    lo = run_scan(arnt, genome_fasta["gz"], min_valid_fraction=0.1, **kw).set_index("start")
    assert hi.loc[500_000, "n_valid"] <= 40
    assert np.isnan(hi.loc[500_000, "ln_tba"])
    assert not np.isnan(lo.loc[500_000, "ln_tba"])


def test_all_n_contig_and_short_contig(arnt, isolated, genome_fasta):
    n = isolated[isolated["chrom"] == "chrN"]
    assert len(n) == (5000 - 200) // 50 + 1
    assert n[COLS].isna().all().all() and (n["n_valid"] == 0).all()
    assert (isolated["chrom"] == "chrShort").sum() == 0
    t = run_scan(arnt, genome_fasta["gz"], W=200, s=50, scores="max", keep_tail=True, region="chrShort")
    assert list(t["start"]) == [0] and list(t["end"]) == [150]


def test_output_order(isolated):
    order = {"chr1": 0, "chr2": 1, "chrN": 2}
    key = isolated["chrom"].map(order).to_numpy() * 10**9 + isolated["start"].to_numpy()
    assert np.all(np.diff(key) > 0)
    assert np.all(isolated["start"] % 50 == 0)
    assert np.all(isolated["end"] - isolated["start"] == 200)


def test_keep_tail(arnt, genome_fasta):
    df = run_scan(arnt, genome_fasta["gz"], W=200, s=50, scores="max", keep_tail=True, region="chr1")
    L = 1_200_123
    assert df["end"].iloc[-1] == L
    assert df["start"].iloc[-1] == ((L - 200) // 50) * 50 + 50
    assert (df["end"] - df["start"] != 200).sum() == 1


def test_lowercase_and_hard_mask(arnt, tmp_path, rng):
    seq = "".join(rng.choice(list("ACGT"), size=5000))
    seq = seq[:1000] + "CACGTG" * 20 + seq[1120:]
    up = tmp_path / "u.fa"
    low = tmp_path / "l.fa"
    write_fasta(up, [("c", seq)])
    write_fasta(low, [("c", seq[:1000] + seq[1000:2000].lower() + seq[2000:])])
    a = run_scan(arnt, up, W=200, s=100)
    b = run_scan(arnt, low, W=200, s=100)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    c = run_scan(arnt, low, W=200, s=100, hard_mask=True)
    assert c.set_index("start").loc[1100, "n_valid"] == 0


def test_reverse_complement_genome(arnt, tmp_path, rng):
    L, W, s = 20_200, 200, 50  # (L − W) mod s = 0, so the window grid maps onto itself
    seq = "".join(rng.choice(list("ACGTN"), size=L, p=[0.24, 0.24, 0.24, 0.24, 0.04]))
    rc = decode(revcomp_ints(encode(seq)))
    f, r = tmp_path / "f.fa", tmp_path / "r.fa"
    write_fasta(f, [("c", seq)])
    write_fasta(r, [("c", rc)])
    a = run_scan(arnt, f, W=W, s=s, pf_mode="isolated", min_valid_fraction=0)
    b = run_scan(arnt, r, W=W, s=s, pf_mode="isolated", min_valid_fraction=0)
    b = b.assign(start=L - b["end"]).sort_values("start").reset_index(drop=True)
    np.testing.assert_allclose(a[COLS].to_numpy(), b[COLS].to_numpy(), rtol=1e-9)


def test_region_matches_full_scan(arnt, genome_fasta, isolated):
    df = run_scan(arnt, genome_fasta["gz"], W=200, s=50, pf_mode="isolated", region="chr2:100,001-150,000")
    full = isolated[(isolated["chrom"] == "chr2") & (isolated["start"] >= 100_000) & (isolated["end"] <= 150_000)]
    pd.testing.assert_frame_equal(df, full.reset_index(drop=True), check_exact=True)


def test_from_genome_background_is_recorded(arnt, genome_fasta, tmp_path):
    out = tmp_path / "s.tsv"
    scan_genome(arnt, genome_fasta["gz"], W=200, s=200, out=str(out), scores="tba", region="chr2")
    head = [l for l in out.read_text().splitlines() if l.startswith("# background=")][0]
    freqs = [float(x.split(":")[1]) for x in head.split("=", 1)[1].split(",")]
    assert freqs[0] == pytest.approx(0.3, abs=0.01) and freqs[1] == pytest.approx(0.2, abs=0.01)


def test_bigwig_round_trip(arnt, genome_fasta, tmp_path):
    pyBigWig = pytest.importorskip("pyBigWig")
    prefix = tmp_path / "scan"
    scan_genome(arnt, genome_fasta["gz"], W=200, s=50, out=str(prefix), fmt="bigwig", scores="tba", region="chr2", background="uniform")
    tsv = run_scan(arnt, genome_fasta["gz"], W=200, s=50, scores="tba", region="chr2")
    bw = pyBigWig.open(f"{prefix}.ln_tba.bw")
    for _, row in tsv.dropna().sample(200, random_state=0).iterrows():
        v = bw.values("chr2", int(row["start"]) + 75, int(row["start"]) + 76)[0]
        assert v == pytest.approx(row["ln_tba"], rel=1e-6)


def test_cli_scan(arnt, genome_fasta, tmp_path):
    from conftest import FIXTURES
    from seqaffinity.cli import main

    out = tmp_path / "cli.tsv"
    main(["scan", str(FIXTURES / "MA0004.1.jaspar"), str(genome_fasta["gz"]), "--window", "200", "--step", "50",
          "--region", "chr2:1-20,000", "--out", str(out), "--background", "uniform", "--pf-isolated"])
    df = pd.read_csv(out, sep="\t", comment="#")
    assert len(df) == (20_000 - 200) // 50 + 1 and list(df.columns[:4]) == ["chrom", "start", "end", "n_valid"]


def test_chunking_is_bit_identical_in_memory(arnt, genome_fasta):
    """Scores 1–4 and isolated PF do not depend on chunk boundaries at all."""
    from seqaffinity.genome import FastaSource, iter_chunks
    from seqaffinity.scan import process_chunk

    em = to_energy(arnt)
    keys = ("max", "tba", "tba_per_window", "trap", "pf")

    def collect(cs):
        src = FastaSource(genome_fasta["gz"])
        out = {}
        for ch in iter_chunks(src, cs, 200, 50, em.width, region="chr1"):
            _, _, _, _, vals = process_chunk(ch, em, np.log(1.2), keys, "both", "isolated", 0.0)
            for k, v in vals.items():
                out.setdefault(k, []).append(v)
        return {k: np.concatenate(v) for k, v in out.items()}

    a, b = collect(77_777), collect(1_000_000)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_bigwig_all_scores_every_window(arnt, genome_fasta, tmp_path):
    """Every window's value sits on its centred step; NA windows are absent; header has all contigs."""
    pyBigWig = pytest.importorskip("pyBigWig")
    W, s = 200, 30  # s does not divide W; (W - s) // 2 = 85
    kw = dict(W=W, s=s, keep_tail=True, background="uniform", pf_mode="isolated")
    prefix = tmp_path / "all"
    scan_genome(arnt, genome_fasta["gz"], out=str(prefix), fmt="bigwig", **kw)
    tsv = run_scan(arnt, genome_fasta["gz"], **kw)
    sizes = {"chr1": 1_200_123, "chr2": 799_877, "chrN": 5_000, "chrShort": 150}
    for col in COLS:
        bw = pyBigWig.open(f"{prefix}.{col}.bw")
        assert bw.chroms() == sizes
        for chrom in ("chr1", "chr2", "chrN", "chrShort"):
            sub = tsv[tsv["chrom"] == chrom]
            if sub.empty:
                continue
            a = sub["start"].to_numpy() + (W - s) // 2
            b = np.minimum(a + s, sub["end"].to_numpy())
            vals = np.array(bw.values(chrom, 0, sizes[chrom], numpy=True))
            got = vals[a[b > a]]
            want = sub[col].to_numpy()[b > a]
            na = np.isnan(want)
            assert np.all(np.isnan(got[na])), (col, chrom)
            np.testing.assert_allclose(got[~na], want[~na].astype(np.float32), rtol=1e-6, err_msg=f"{col} {chrom}")
        bw.close()


def test_cli_bigwig(arnt, genome_fasta, tmp_path):
    pyBigWig = pytest.importorskip("pyBigWig")
    from conftest import FIXTURES
    from seqaffinity.cli import main

    prefix = tmp_path / "cli"
    main(["scan", str(FIXTURES / "MA0004.1.jaspar"), str(genome_fasta["gz"]), "--format", "bigwig",
          "--out", str(prefix) + ".bw", "--scores", "trap", "--region", "chr2:1-50,000"])
    for col in ("trap", "trap_per_bp"):
        bw = pyBigWig.open(f"{prefix}.{col}.bw")
        assert list(bw.chroms()) == ["chr2"] and bw.header()["nBasesCovered"] > 0
    assert (tmp_path / "cli.params.txt").read_text().startswith("# motif=")
