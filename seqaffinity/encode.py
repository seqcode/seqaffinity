"""Sequence encoding: ACGT → 0..3, everything else → -1."""

from __future__ import annotations

import numpy as np

_IUPAC = "RYSWKMBDHVN"


def _make_lut(hard_mask: bool) -> np.ndarray:
    lut = np.full(256, -2, dtype=np.int8)  # -2 = invalid character
    for i, b in enumerate("ACGT"):
        lut[ord(b)] = i
        lut[ord(b.lower())] = -1 if hard_mask else i
    for c in _IUPAC + "U":
        lut[ord(c)] = -1
        lut[ord(c.lower())] = -1
    lut[ord("U")] = 3
    lut[ord("u")] = -1 if hard_mask else 3
    for c in ".-":
        lut[ord(c)] = -1
    return lut


_LUT = _make_lut(False)
_LUT_HARD = _make_lut(True)


def encode(seq, hard_mask: bool = False) -> np.ndarray:
    """Encode a nucleotide sequence as an int8 array.

    Parameters
    ----------
    seq : str, bytes or np.ndarray of uint8
        Sequence. ``ACGT`` (either case; ``U`` read as ``T``) map to 0..3; IUPAC
        ambiguity codes, ``N``, ``.`` and ``-`` map to -1.
    hard_mask : bool
        Treat lowercase (soft-masked) bases as ambiguous (-1).

    Returns
    -------
    np.ndarray
        int8 array of the same length.

    Raises
    ------
    ValueError
        If the sequence contains characters that are not nucleotide codes.
    """
    if isinstance(seq, str):
        raw = np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)
    elif isinstance(seq, (bytes, bytearray, memoryview)):
        raw = np.frombuffer(seq, dtype=np.uint8)
    else:
        raw = np.asarray(seq, dtype=np.uint8)
    out = (_LUT_HARD if hard_mask else _LUT)[raw]
    if out.size and out.min() == -2:
        bad = sorted({chr(c) for c in raw[out == -2][:20]})
        raise ValueError(f"invalid nucleotide character(s): {bad}")
    return out


def revcomp_ints(arr: np.ndarray) -> np.ndarray:
    """Reverse complement of an encoded sequence (ambiguity codes stay -1)."""
    arr = np.asarray(arr, dtype=np.int8)
    out = np.where(arr >= 0, 3 - arr, arr).astype(np.int8)
    return out[::-1].copy()


def decode(arr: np.ndarray) -> str:
    """Inverse of :func:`encode` (ambiguity codes become ``N``)."""
    arr = np.asarray(arr)
    return np.frombuffer(b"ACGTN", dtype="S1")[np.where(arr < 0, 4, arr)].tobytes().decode()


def base_composition(arrays) -> np.ndarray:
    """Count A, C, G, T over one or more encoded arrays (ambiguity codes ignored)."""
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    total = np.zeros(4, dtype=np.int64)
    for a in arrays:
        a = np.asarray(a)
        total += np.bincount(a[a >= 0].astype(np.int64), minlength=4)[:4]
    return total
