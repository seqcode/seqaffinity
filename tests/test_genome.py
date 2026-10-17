import gzip

import numpy as np
import pytest

from conftest import write_fasta
from seqaffinity.encode import decode
from seqaffinity.genome import FastaSource, iter_chunks, iter_fasta_bytes, parse_region, read_bed_sequences, window_starts


def test_streaming_reader_handles_quirks(tmp_path):
    p = tmp_path / "q.fa"
    p.write_bytes(b">a desc\r\nACGT\r\nAC\r\nGTACGT\r\n>b\nnnACgt\n\n>c\n")
    recs = list(iter_fasta_bytes(p))
    assert recs == [("a", b"ACGTACGTACGT"), ("b", b"nnACgt"), ("c", b"")]
    gz = tmp_path / "q.fa.gz"
    gz.write_bytes(gzip.compress(p.read_bytes()))
    assert list(iter_fasta_bytes(gz)) == recs


def test_duplicate_names_raise(tmp_path):
    p = tmp_path / "d.fa"
    p.write_text(">a\nACGT\n>a\nACGT\n")
    with pytest.raises(ValueError):
        list(iter_fasta_bytes(p))
    with pytest.raises(ValueError):
        list(FastaSource(p, backend="stream").contigs())


def test_backends_agree(genome_fasta, genome_records):
    pytest.importorskip("pyfaidx")
    a = FastaSource(genome_fasta["plain"], backend="pyfaidx")
    b = FastaSource(genome_fasta["gz"])
    assert a.backend == "pyfaidx" and b.backend == "stream"
    for ca, cb, (name, seq) in zip(a.contigs(), b.contigs(), genome_records):
        assert ca.name == cb.name == name and ca.length == cb.length == len(seq)
        np.testing.assert_array_equal(ca.fetch(1000, 3000), cb.fetch(1000, 3000))
        assert decode(ca.fetch(0, 50)) == seq[:50].upper()


def test_mixed_line_widths_fall_back_to_streaming(tmp_path):
    p = tmp_path / "m.fa"
    p.write_text(">a\nACGTACGT\nACG\nACGTACGT\n")
    with pytest.warns(UserWarning):
        src = FastaSource(p)
    assert src.backend == "stream"
    assert decode(next(src.contigs()).fetch(0, 19)) == "ACGTACGTACGACGTACGT"


def test_parse_region():
    assert parse_region("chr1") == ("chr1", 0, None)
    assert parse_region("chr1:1-10,000,000") == ("chr1", 0, 10_000_000)
    with pytest.raises(ValueError):
        parse_region("chr1:0-5")


@pytest.mark.parametrize(
    "L,W,s,keep,expected",
    [
        (1000, 200, 50, False, (list(range(0, 801, 50)), 200)),
        (1010, 200, 50, True, (list(range(0, 801, 50)) + [850], None)),  # tail [850, 1010)
        (1000, 200, 50, True, (list(range(0, 801, 50)), 200)),  # last window ends at L: no tail
        (150, 200, 50, True, ([0], None)),  # contig shorter than W
        (150, 200, 50, False, ([], None)),
        (1010, 200, 30, True, (list(range(0, 781, 30)) + [810], None)),  # (L-W) mod s = 0 → no extra? see below
    ],
)
def test_window_starts(L, W, s, keep, expected):
    starts, lengths = window_starts(L, W, s, 6, keep)
    if (L, s) == (1010, 30):
        # 810 = 27·30 is itself a full window start (810 + 200 = 1010): no tail is added.
        assert starts[-1] == 810 and lengths[-1] == 200
        return
    assert list(starts) == expected[0]
    assert np.all(starts + lengths <= L)
    if keep and expected[0] and expected[0][-1] + 200 > L:
        assert starts[-1] + lengths[-1] == L


def test_chunks_cover_each_window_once(genome_fasta):
    src = FastaSource(genome_fasta["gz"])
    for cs in (100_000, 333_333):
        seen = []
        for ch in iter_chunks(src, cs, 200, 50, 6, overlap=37):
            seen.extend(zip([ch.chrom] * len(ch.starts), ch.starts))
            assert ch.seq.shape[0] >= (ch.starts + ch.lengths).max() - ch.seq_start
            assert ch.seq_start <= ch.starts.min()
        assert len(seen) == len(set(seen))
        assert len([1 for c, _ in seen if c == "chr1"]) == (1_200_123 - 200) // 50 + 1


def test_read_bed_sequences(tmp_path, genome_fasta, genome_records):
    bed = tmp_path / "x.bed"
    bed.write_text("chr2\t100\t110\tpeakA\nchr1\t5\t15\n")
    out = read_bed_sequences(genome_fasta["gz"], bed)
    seqs = dict(genome_records)
    assert out == [("peakA", seqs["chr2"][100:110].upper()), ("chr1:5-15", seqs["chr1"][5:15].upper())]
