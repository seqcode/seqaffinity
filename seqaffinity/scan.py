"""Genome-wide window scanning: chunk → streams → aggregate → ordered writer."""

from __future__ import annotations

import collections
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .aggregate import aggregate
from .api import _as_pwm, run_params
from .genome import Chunk, FastaSource, iter_chunks, parse_region
from .params import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MIN_VALID_FRACTION,
    SCORE_KEYS,
    normalise_scores,
    score_columns,
    trap_r0,
    validate_r0,
    validate_scan,
    validate_strands,
)
from .pf import ln_activity, pf_occupancy_batch
from .pwm import EnergyMatrix, to_energy
from .streams import streams
from .windows import valid_windows, window_energies
from .writers import make_writer

_ISOLATED_BATCH = 4096


def _pf_isolated(ln_a: np.ndarray, idx: np.ndarray, n_motif: np.ndarray, w: int) -> np.ndarray:
    """Partition function run separately inside every window (vectorised across windows)."""
    out = np.zeros(idx.shape[0])
    for n in np.unique(n_motif):
        sel = np.flatnonzero(n_motif == n)
        view = sliding_window_view(ln_a, int(n))
        for b in range(0, sel.size, _ISOLATED_BATCH):
            part = sel[b : b + _ISOLATED_BATCH]
            occ, _ = pf_occupancy_batch(view[idx[part]], w)
            out[part] = occ.sum(axis=1)
    return out


def process_chunk(
    chunk: Chunk,
    em: EnergyMatrix,
    ln_r0: float,
    scores: tuple[str, ...],
    strands: str,
    pf_mode: str,
    min_valid_fraction: float,
    pf_backend: str = "auto",
):
    """Score every window of one chunk.

    Returns
    -------
    tuple
        ``(chrom, starts, ends, n_valid, values)`` with ``values`` a dict of score
        columns (NaN where the window fails ``min_valid_fraction``).
    """
    w = em.width
    E_win = window_energies(chunk.seq, em, strands=strands)
    valid = valid_windows(chunk.seq, w)
    want = {"valid"}
    if "max" in scores:
        want.add("best")
    if {"tba", "tba_per_window"} & set(scores):
        want.add("k")
    if "trap" in scores:
        want.add("occ")
    if "pf" in scores and pf_mode == "genome":
        want.add("pf_occ")
    st = streams(E_win, em, ln_r0, want, valid=valid, pf_backend=pf_backend)
    idx = chunk.starts - chunk.seq_start
    n_motif = chunk.lengths - w + 1
    agg = aggregate(st, idx, n_motif)

    n_strands = 1 if strands == "plus" else 2
    vals: dict[str, np.ndarray] = {}
    if "max" in scores:
        vals["max_score"] = agg["best"]
    if "tba" in scores:
        with np.errstate(divide="ignore"):
            vals["ln_tba"] = np.log(agg["k"])
    if "tba_per_window" in scores:
        vals["tba_per_window"] = agg["k"] / (n_strands * n_motif)
    if "trap" in scores:
        vals["trap"] = agg["occ"]
        vals["trap_per_bp"] = agg["occ"] / chunk.lengths
    if "pf" in scores:
        if pf_mode == "genome":
            vals["partition_fn"] = agg["pf_occ"]
        else:
            vals["partition_fn"] = _pf_isolated(ln_activity(E_win, ln_r0), idx, n_motif, w)
    n_valid = agg["valid"]
    na = n_valid < min_valid_fraction * n_motif
    if na.any():
        for v in vals.values():
            v[na] = np.nan
    return chunk.chrom, chunk.starts, chunk.starts + chunk.lengths, n_valid, vals


def scan_genome(
    pwm,
    fasta_path,
    W: int = 200,
    s: int = 50,
    out="-",
    fmt: str = "tsv",
    workers: int = 1,
    lam: float = 0.7,
    pseudocount: float | str = "auto",
    background="from-genome",
    R0: float | None = None,
    strands: str = "both",
    scores=SCORE_KEYS,
    pf_mode: str = "genome",
    pf_overlap: int | None = None,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    keep_tail: bool = False,
    hard_mask: bool = False,
    region: str | None = None,
    fasta_backend: str = "auto",
    pf_backend: str = "auto",
) -> dict:
    """Score every window ``[k·s, k·s + W)`` of a genome and stream the result.

    Parameters
    ----------
    pwm : PWM, EnergyMatrix, path or (w, 4) array
    fasta_path : str
        Genome FASTA (plain, gzip or BGZF).
    W, s : int
        Window size and step (``1 ≤ s ≤ W``, ``W ≥ w``).
    out : str or file-like
        Output path (``"-"`` = stdout). For bigWig, a prefix: one
        ``{prefix}.{column}.bw`` per score.
    fmt : {"tsv", "bed", "bigwig"}
    workers : int
        Worker processes (chunks are written in genomic order).
    lam, pseudocount, R0, strands
        See :func:`seqaffinity.api.score_sequences`.
    background : "from-genome", "uniform", "a,c,g,t" or array-like
        ``from-genome`` counts the scanned sequence (the region, if given) first.
    scores : iterable of str
    pf_mode : {"genome", "isolated"}
        Genome-context occupancy or a per-window partition function.
    pf_overlap : int or None
        Context flank for genome-mode PF (default ``64·w``).
    min_valid_fraction : float
        Windows with fewer valid motif windows than this fraction are NA.
    chunk_size : int
        Window starts per chunk block, in bases.
    keep_tail : bool
        Add one shorter window at each contig end.
    hard_mask : bool
        Treat lowercase as N.
    region : str or None
        ``chr`` or ``chr:start-end`` (1-based inclusive).
    fasta_backend, pf_backend : str
        Implementation choices (mainly for testing).

    Returns
    -------
    dict
        Summary: number of windows and chunks, and the run parameters.
    """
    scores = normalise_scores(scores)
    strands = validate_strands(strands)
    if pf_mode not in ("genome", "isolated"):
        raise ValueError(f"pf_mode must be 'genome' or 'isolated', got {pf_mode!r}")
    source = FastaSource(fasta_path, hard_mask=hard_mask, backend=fasta_backend)

    regions = None
    if region:
        chrom, lo, hi = parse_region(region)
        regions = {chrom: (lo, hi)}

    need_comp = isinstance(background, str) and background == "from-genome"
    chrom_sizes = None
    if isinstance(pwm, EnergyMatrix):
        em = pwm
        need_comp = False
    else:
        p = _as_pwm(pwm)
        if need_comp or fmt == "bigwig":
            chrom_sizes, counts = source.sizes_and_composition(regions, composition=need_comp)
        if need_comp:
            if counts.sum() == 0:
                raise ValueError("cannot estimate background: genome has no ACGT bases")
            background = (counts + 1) / (counts.sum() + 4)
        em = to_energy(p, lam=lam, pseudocount=pseudocount, background=background)
    if fmt == "bigwig" and chrom_sizes is None:
        chrom_sizes, _ = source.sizes_and_composition(regions, composition=False)

    w = em.width
    validate_scan(W, s, w, min_valid_fraction, chunk_size)
    r0 = trap_r0(w) if R0 is None else validate_r0(R0)
    ln_r0 = float(np.log(r0))
    overlap = 0
    if "pf" in scores and pf_mode == "genome":
        overlap = 64 * w if pf_overlap is None else int(pf_overlap)

    params = run_params(
        em,
        r0,
        strands,
        window=W,
        step=s,
        pf_mode=pf_mode,
        pf_overlap=overlap if pf_mode == "genome" else "n/a",
        min_valid_fraction=min_valid_fraction,
        keep_tail=keep_tail,
        hard_mask=hard_mask,
        region=region or "all",
        genome=os.fspath(fasta_path),
    )
    columns = score_columns(scores)
    writer = make_writer(fmt, out, columns, params, chrom_sizes=chrom_sizes, W=W, s=s)

    chunks = iter_chunks(source, chunk_size, W, s, w, overlap=overlap, keep_tail=keep_tail, region=region)
    args = (em, ln_r0, scores, strands, pf_mode, min_valid_fraction, pf_backend)
    n_windows = 0
    n_chunks = 0
    try:
        if workers <= 1:
            for ch in chunks:
                res = process_chunk(ch, *args)
                writer.write(*res)
                n_windows += res[1].shape[0]
                n_chunks += 1
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                pending: collections.deque = collections.deque()
                for ch in chunks:
                    pending.append(ex.submit(process_chunk, ch, *args))
                    while len(pending) >= 2 * workers:
                        res = pending.popleft().result()
                        writer.write(*res)
                        n_windows += res[1].shape[0]
                        n_chunks += 1
                while pending:
                    res = pending.popleft().result()
                    writer.write(*res)
                    n_windows += res[1].shape[0]
                    n_chunks += 1
    finally:
        writer.close()
    return {"n_windows": n_windows, "n_chunks": n_chunks, "fasta_backend": source.backend, **params}
