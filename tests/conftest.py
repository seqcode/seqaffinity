import gzip
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def random_seq(rng, n, p_ambig=0.0, alphabet="ACGT"):
    s = rng.choice(list(alphabet), size=n)
    if p_ambig:
        s[rng.random(n) < p_ambig] = "N"
    return "".join(s)


def random_counts(rng, w, depth=50):
    """Random count matrix with a clear consensus at each position."""
    c = rng.dirichlet(np.full(4, 0.5), size=w) * depth
    return np.round(c)


def write_fasta(path, records, width=60, opener=open):
    with opener(path, "wt") as fh:
        for name, seq in records:
            fh.write(f">{name} test record\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i : i + width] + "\n")


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def arnt_path():
    return FIXTURES / "MA0004.1.jaspar"


def _make_genome(seed=7):
    """~2 Mb synthetic genome with planted sites, N runs, soft-masking and odd contigs."""
    rng = np.random.default_rng(seed)
    recs = []
    for name, L in (("chr1", 1_200_123), ("chr2", 799_877)):
        s = rng.choice(np.array(list("ACGT")), size=L, p=[0.3, 0.2, 0.2, 0.3])
        # Planted Arnt sites (CACGTG) and near-sites.
        for pos in rng.integers(0, L - 10, size=L // 2000):
            s[pos : pos + 6] = list("CACGTG" if rng.random() < 0.7 else "CACGTA")
        # N runs of various lengths.
        for pos, ln in zip(rng.integers(0, L - 5000, size=20), rng.integers(1, 3000, size=20)):
            s[pos : pos + ln] = "N"
        # A known 150 bp N run at a window boundary-friendly place.
        s[500_020 : 500_170] = "N"
        # Soft-masked stretches.
        for pos, ln in zip(rng.integers(0, L - 5000, size=30), rng.integers(10, 4000, size=30)):
            s[pos : pos + ln] = np.char.lower(s[pos : pos + ln])
        recs.append((name, "".join(s)))
    recs.append(("chrN", "N" * 5_000))
    recs.append(("chrShort", "".join(rng.choice(list("ACGT"), size=150))))
    return recs


@pytest.fixture(scope="session")
def genome_records():
    return _make_genome()


@pytest.fixture(scope="session")
def genome_fasta(tmp_path_factory, genome_records):
    d = tmp_path_factory.mktemp("genome")
    plain = d / "genome.fa"
    write_fasta(plain, genome_records)
    gz = d / "genome.fa.gz"
    write_fasta(gz, genome_records, width=70, opener=gzip.open)
    return {"plain": plain, "gz": gz, "dir": d}
