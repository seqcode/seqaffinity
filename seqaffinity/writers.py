"""Streaming output writers for genome scans: TSV, BED-like and bigWig."""

from __future__ import annotations

import io
import sys

import numpy as np

FORMATS = ("tsv", "bed", "bigwig")


def _header_lines(params: dict) -> str:
    return "".join(f"# {k}={v}\n" for k, v in params.items())


class TableWriter:
    """TSV (header row) or BED-like (commented header row) output.

    Parameters
    ----------
    out : str or file-like
        Path, ``"-"`` for stdout, or an open text file.
    columns : list of str
        Score column names.
    params : dict
        Written as ``# key=value`` lines.
    fmt : {"tsv", "bed"}
    """

    def __init__(self, out, columns, params: dict, fmt: str = "tsv"):
        self.columns = list(columns)
        self._own = False
        if out == "-" or out is None:
            self.fh = sys.stdout
        elif isinstance(out, (str, bytes)) or hasattr(out, "__fspath__"):
            self.fh = open(out, "w")
            self._own = True
        else:
            self.fh = out
        names = ["chrom", "start", "end", "n_valid", *self.columns]
        self.fh.write(_header_lines(params))
        self.fh.write(("#" if fmt == "bed" else "") + "\t".join(names) + "\n")

    def write(self, chrom: str, starts, ends, n_valid, values: dict[str, np.ndarray]) -> None:
        import pandas as pd

        df = pd.DataFrame({"chrom": chrom, "start": starts, "end": ends, "n_valid": n_valid})
        for c in self.columns:
            df[c] = values[c]
        buf = io.StringIO()
        df.to_csv(buf, sep="\t", header=False, index=False, float_format="%.10g", na_rep="NA")
        self.fh.write(buf.getvalue())

    def close(self) -> None:
        if self._own:
            self.fh.close()
        else:
            self.fh.flush()


class BigWigWriter:
    """One bigWig file per score column: ``{prefix}.{column}.bw``.

    bigWig cannot hold overlapping intervals, so each window's value is written
    to its centred step ``[start + (W − s)//2, + s)``, clipped to the contig. NA
    windows are omitted. Values are stored as float32.
    """

    def __init__(self, prefix: str, columns, chrom_sizes: list[tuple[str, int]], W: int, s: int, params: dict):
        try:
            import pyBigWig
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("bigWig output needs pyBigWig (pip install 'seqaffinity[bigwig]')") from exc
        prefix = str(prefix)
        if prefix.endswith(".bw"):
            prefix = prefix[:-3]
        self.columns = list(columns)
        self.W, self.s = W, s
        self.sizes = dict(chrom_sizes)
        self.paths = {c: f"{prefix}.{c}.bw" for c in self.columns}
        self.files = {}
        for c, path in self.paths.items():
            bw = pyBigWig.open(path, "w")
            bw.addHeader(list(chrom_sizes), maxZooms=10)
            self.files[c] = bw
        with open(f"{prefix}.params.txt", "w") as fh:
            fh.write(_header_lines(params))
        self._last_end: dict[str, int] = {}

    def steps(self, chrom: str, starts: np.ndarray, ends: np.ndarray):
        off = (self.W - self.s) // 2
        a = starts + off
        b = np.minimum(a + self.s, ends)
        b = np.minimum(b, self.sizes[chrom])
        return a, b

    def write(self, chrom: str, starts, ends, n_valid, values: dict[str, np.ndarray]) -> None:
        a, b = self.steps(chrom, np.asarray(starts), np.asarray(ends))
        ok_geom = b > a
        for c in self.columns:
            v = np.asarray(values[c], dtype=np.float64)
            ok = ok_geom & np.isfinite(v)
            if not ok.any():
                continue
            self.files[c].addEntries(
                [chrom] * int(ok.sum()),
                a[ok].astype(np.int64).tolist(),
                ends=b[ok].astype(np.int64).tolist(),
                values=v[ok].tolist(),
            )

    def close(self) -> None:
        for bw in self.files.values():
            bw.close()


def make_writer(fmt: str, out, columns, params: dict, chrom_sizes=None, W=None, s=None):
    """Create a writer for ``fmt`` in :data:`FORMATS`."""
    if fmt in ("tsv", "bed"):
        return TableWriter(out, columns, params, fmt)
    if fmt == "bigwig":
        return BigWigWriter(out, columns, chrom_sizes, W, s, params)
    raise ValueError(f"unknown output format {fmt!r}; choose from {FORMATS}")
