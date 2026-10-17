"""FASTA access (indexed or streaming), region parsing and chunk iteration for scans."""

from __future__ import annotations

import gzip
import os
import re
import warnings
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from .encode import encode

_WS = b" \t\r\n"


# --------------------------------------------------------------------------- FASTA readers


def _is_gzip(path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def _open_bytes(path):
    return gzip.open(path, "rb") if _is_gzip(path) else open(path, "rb")


def iter_fasta_bytes(path) -> Iterator[tuple[str, bytes]]:
    """Stream ``(name, sequence bytes)`` records from a FASTA file.

    Handles plain, gzip and BGZF files, mixed line widths and CRLF line endings.
    The name is the first whitespace-delimited token of the header. Duplicate
    names raise ``ValueError``.
    """
    seen: set[str] = set()
    name = None
    buf = bytearray()
    with _open_bytes(path) as fh:
        for line in fh:
            if line.startswith(b">"):
                if name is not None:
                    yield name, bytes(buf).translate(None, _WS)
                toks = line[1:].split()
                name = toks[0].decode() if toks else ""
                if name in seen:
                    raise ValueError(f"duplicate sequence name {name!r} in {path}")
                seen.add(name)
                buf = bytearray()
            elif name is not None:
                buf += line
            elif line.strip() and not line.startswith(b";"):
                raise ValueError(f"{path}: sequence data before the first '>' header")
    if name is not None:
        yield name, bytes(buf).translate(None, _WS)


def read_fasta(path) -> list[tuple[str, str]]:
    """Read a (small) FASTA file into ``[(name, sequence), ...]``."""
    return [(n, s.decode("ascii")) for n, s in iter_fasta_bytes(path)]


# --------------------------------------------------------------------------- sources


@dataclass
class Contig:
    """A chromosome handle: ``fetch(start, end)`` returns the encoded slice."""

    name: str
    length: int
    fetch: Callable[[int, int], np.ndarray]


class FastaSource:
    """Genome access for scanning.

    Parameters
    ----------
    path : str
        FASTA file (plain, gzip or BGZF).
    hard_mask : bool
        Encode lowercase bases as ambiguous.
    backend : {"auto", "pyfaidx", "stream"}
        ``"auto"`` uses pyfaidx for uncompressed files when it is installed (fast
        region access, writes a ``.fai`` next to the file), and otherwise streams
        the file one chromosome at a time.
    """

    def __init__(self, path, hard_mask: bool = False, backend: str = "auto"):
        self.path = os.fspath(path)
        self.hard_mask = hard_mask
        self._fa = None
        if backend not in ("auto", "pyfaidx", "stream"):
            raise ValueError(f"unknown FASTA backend {backend!r}")
        if backend in ("auto", "pyfaidx") and not _is_gzip(self.path):
            try:
                import pyfaidx

                self._fa = pyfaidx.Fasta(self.path, as_raw=True, sequence_always_upper=False, rebuild=False)
            except ImportError:
                if backend == "pyfaidx":
                    raise
            except Exception as exc:  # malformed for faidx: mixed line widths, duplicates, ...
                if backend == "pyfaidx":
                    raise
                if "duplicate" in str(exc).lower():
                    raise ValueError(str(exc)) from exc
                warnings.warn(f"pyfaidx could not index {self.path} ({exc}); streaming instead", stacklevel=2)
                self._fa = None
        elif backend == "pyfaidx":
            raise ValueError("pyfaidx backend needs an uncompressed FASTA; bgzip support would require Biopython")
        self.backend = "pyfaidx" if self._fa is not None else "stream"

    def contigs(self, names: set[str] | None = None) -> Iterator[Contig]:
        """Yield contigs in file order (restricted to ``names`` if given)."""
        if self._fa is not None:
            fa = self._fa
            for name in fa.keys():
                if names is not None and name not in names:
                    continue
                rec = fa[name]
                yield Contig(name, len(rec), lambda a, b, rec=rec: encode(rec[a:b], hard_mask=self.hard_mask))
        else:
            for name, raw in iter_fasta_bytes(self.path):
                if names is not None and name not in names:
                    continue
                arr = encode(raw, hard_mask=self.hard_mask)
                yield Contig(name, arr.shape[0], lambda a, b, arr=arr: arr[a:b])

    def sizes_and_composition(self, regions: dict[str, tuple[int, int]] | None = None, composition: bool = True):
        """One pass over the genome: contig sizes and A/C/G/T counts.

        Returns
        -------
        sizes : list of (name, length)
        counts : np.ndarray (4,) or None
        """
        sizes = []
        total = np.zeros(4, dtype=np.int64) if composition else None
        if not composition and self._fa is not None:
            return [(n, len(self._fa[n])) for n in self._fa.keys() if regions is None or n in regions], None
        for c in self.contigs(set(regions) if regions else None):
            sizes.append((c.name, c.length))
            if composition:
                lo, hi = (0, c.length) if regions is None else _clip(regions[c.name], c.length)
                step = 20_000_000
                for a in range(lo, hi, step):
                    seg = c.fetch(a, min(hi, a + step))
                    total += np.bincount(seg[seg >= 0].astype(np.int64), minlength=4)[:4]
        return sizes, total


def _clip(region: tuple[int, int], length: int) -> tuple[int, int]:
    lo, hi = region
    return max(0, lo), min(length, hi if hi is not None else length)


def parse_region(region: str) -> tuple[str, int, int | None]:
    """Parse ``chr1``, ``chr1:1-10,000,000`` (1-based, inclusive) into 0-based half-open."""
    m = re.fullmatch(r"([^:]+)(?::([\d,]+)-([\d,]+))?", region.strip())
    if not m:
        raise ValueError(f"cannot parse region {region!r}")
    chrom = m.group(1)
    if m.group(2) is None:
        return chrom, 0, None
    start = int(m.group(2).replace(",", ""))
    end = int(m.group(3).replace(",", ""))
    if start < 1 or end < start:
        raise ValueError(f"invalid region {region!r}")
    return chrom, start - 1, end


# --------------------------------------------------------------------------- windows and chunks


def window_starts(L: int, W: int, s: int, w: int, keep_tail: bool, lo: int = 0, hi: int | None = None):
    """Window starts and lengths for a contig of length ``L``.

    Full windows are ``[k·s, k·s + W)`` with ``k·s + W ≤ L``. With ``keep_tail``,
    one extra window ``[a_last + s, L)`` (or ``[0, L)`` when ``L < W``) is added if
    the last full window does not reach ``L`` and the tail is at least ``w`` long.
    Only windows inside ``[lo, hi)`` are returned; the tail is kept only when
    ``hi`` is the contig end.

    Returns
    -------
    starts, lengths : np.ndarray (int64)
    """
    hi = L if hi is None else min(hi, L)
    n_full = (L - W) // s + 1 if L >= W else 0
    starts = np.arange(n_full, dtype=np.int64) * s
    lengths = np.full(n_full, W, dtype=np.int64)
    if keep_tail and hi == L:
        t = (starts[-1] + s) if n_full else 0
        if (not n_full or starts[-1] + W < L) and L - t >= w:
            starts = np.append(starts, t)
            lengths = np.append(lengths, L - t)
    keep = (starts >= lo) & (starts + lengths <= hi)
    return starts[keep], lengths[keep]


@dataclass
class Chunk:
    """One unit of scan work.

    Attributes
    ----------
    chrom : str
    seq_start : int
        Genomic coordinate of ``seq[0]``.
    seq : np.ndarray
        Encoded sequence covering all windows plus context flanks.
    starts, lengths : np.ndarray
        Genomic windows emitted by this chunk.
    index : int
        Running chunk number (output order).
    """

    chrom: str
    seq_start: int
    seq: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    index: int = 0


def iter_chunks(
    source: FastaSource,
    chunk_size: int,
    W: int,
    s: int,
    w: int,
    overlap: int = 0,
    keep_tail: bool = False,
    region: str | None = None,
) -> Iterator[Chunk]:
    """Split the genome into chunks of windows whose starts fall in consecutive
    ``chunk_size`` blocks.

    Each chunk's sequence spans its windows plus ``overlap`` bases of context on
    each side (clipped to the contig), so the result does not depend on
    ``chunk_size`` except through that context.
    """
    names = None
    lo_r, hi_r = 0, None
    if region:
        rchrom, lo_r, hi_r = parse_region(region)
        names = {rchrom}
    idx = 0
    found = False
    for contig in source.contigs(names):
        found = True
        L = contig.length
        starts, lengths = window_starts(L, W, s, w, keep_tail, lo_r, hi_r)
        if starts.size == 0:
            continue
        block = starts // chunk_size
        cuts = np.flatnonzero(np.diff(block)) + 1
        for st, ln in zip(np.split(starts, cuts), np.split(lengths, cuts)):
            a = max(0, int(st[0]) - overlap)
            b = min(L, int((st + ln).max()) + overlap)
            yield Chunk(contig.name, a, contig.fetch(a, b), st, ln, idx)
            idx += 1
    if region and not found:
        raise ValueError(f"region contig {names} not found in {source.path}")


# --------------------------------------------------------------------------- BED extraction


def read_bed_sequences(fasta_path, bed_path, hard_mask: bool = False) -> list[tuple[str, str]]:
    """Extract sequences for BED intervals (0-based, half-open) from a FASTA.

    Returns ``[(name, sequence), ...]`` in BED order. Intervals that run off a
    contig are clipped; intervals on missing contigs are skipped. Names come from
    BED column 4 if present, else ``chrom:start-end``. Minus-strand intervals are
    not reverse-complemented: all scores are strand-symmetric.
    """
    from .encode import decode

    entries: list[tuple[str, int, int, str]] = []
    with open(bed_path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            chrom, a, b = f[0], int(f[1]), int(f[2])
            name = f[3] if len(f) > 3 and f[3] not in ("", ".") else f"{chrom}:{a}-{b}"
            entries.append((chrom, a, b, name))
    by_chrom: dict[str, list[int]] = {}
    for i, e in enumerate(entries):
        by_chrom.setdefault(e[0], []).append(i)
    seqs: dict[int, str] = {}
    for c in FastaSource(fasta_path, hard_mask=hard_mask).contigs(set(by_chrom)):
        for i in by_chrom[c.name]:
            _, a, b, _ = entries[i]
            seqs[i] = decode(c.fetch(max(0, a), min(c.length, b)))
    return [(entries[i][3], seqs[i]) for i in range(len(entries)) if i in seqs]
