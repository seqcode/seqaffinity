"""Score a set of sequences under all five rules."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from .encode import base_composition, encode
from .params import SCORE_KEYS, normalise_scores, score_columns, validate_r0, validate_strands
from .pf import ln_activity, pf_occupancy_batch
from .pwm import PWM, EnergyMatrix, load_pwm, to_energy
from .scores import max_site, resolve_r0, split_strands, tba, tba_per_window, trap
from .windows import window_energies

_PF_BATCH = 2048


def _as_pwm(pwm) -> PWM:
    if isinstance(pwm, PWM):
        return pwm
    if isinstance(pwm, (str, os.PathLike)):
        return load_pwm(pwm)
    return PWM.from_matrix(pwm)


def _as_records(seqs) -> list[tuple[str, str]]:
    """Normalise sequences to ``[(id, seq), ...]``."""
    if isinstance(seqs, (str, os.PathLike)) and not (isinstance(seqs, str) and set(seqs.upper()) <= set("ACGTNRYSWKMBDHVU.-")):
        from .genome import read_fasta

        return read_fasta(seqs)
    if isinstance(seqs, str):
        return [("seq1", seqs)]
    if isinstance(seqs, dict):
        return [(str(k), v) for k, v in seqs.items()]
    out = []
    for i, s in enumerate(seqs):
        if isinstance(s, tuple):
            out.append((str(s[0]), s[1]))
        else:
            out.append((f"seq{i + 1}", s))
    return out


def resolve_energy(
    pwm,
    lam: float = 0.7,
    pseudocount: float | str = "auto",
    background="uniform",
    encoded: Iterable[np.ndarray] | None = None,
) -> EnergyMatrix:
    """Build the energy matrix, estimating the background from ``encoded`` if requested."""
    if isinstance(pwm, EnergyMatrix):
        return pwm
    if isinstance(background, str) and background in ("from-seqs", "from-genome"):
        if encoded is None:
            raise ValueError(f"background={background!r} needs sequences")
        counts = base_composition(list(encoded))
        if counts.sum() == 0:
            raise ValueError("cannot estimate background: no ACGT bases")
        background = (counts + 1) / (counts.sum() + 4)
    return to_energy(_as_pwm(pwm), lam=lam, pseudocount=pseudocount, background=background)


def score_one(
    pwm,
    seq: str,
    lam: float = 0.7,
    pseudocount: float | str = "auto",
    background="uniform",
    R0: float | None = None,
    strands: str = "both",
    hard_mask: bool = False,
) -> dict:
    """Score a single sequence and return every :class:`ScoreResult` plus energies.

    Returns
    -------
    dict
        Keys ``max``, ``tba``, ``tba_per_window``, ``trap``, ``pf`` (ScoreResult),
        and ``E_win`` (``(2, n_win)`` energies).
    """
    from .scores import partition_fn

    s = encode(seq, hard_mask=hard_mask)
    em = resolve_energy(pwm, lam, pseudocount, background, encoded=[s])
    E_win = window_energies(s, em, strands=validate_strands(strands))
    r0 = resolve_r0(R0, em)
    return {
        "max": max_site(E_win, em),
        "tba": tba(E_win),
        "tba_per_window": tba_per_window(E_win),
        "trap": trap(E_win, em, r0),
        "pf": partition_fn(E_win, em, r0),
        "E_win": E_win,
    }


def score_sequences(
    pwm,
    seqs,
    lam: float = 0.7,
    pseudocount: float | str = "auto",
    background="uniform",
    R0: float | None = None,
    strands: str = "both",
    scores=SCORE_KEYS,
    return_profiles: bool = False,
    hard_mask: bool = False,
):
    """Score many sequences under the selected rules.

    Parameters
    ----------
    pwm : PWM, EnergyMatrix, path or (w, 4) array
        The motif. An :class:`EnergyMatrix` is used as is (``lam``,
        ``pseudocount`` and ``background`` are then ignored).
    seqs : FASTA path, str, list of str, list of (id, seq) or dict
    lam, pseudocount, background
        See :func:`seqaffinity.pwm.to_energy`. ``background`` may also be
        ``"from-seqs"``.
    R0 : float or None
        Effective concentration for TRAP and the partition function. Defaults to
        TRAP's width formula.
    strands : {"both", "plus"}
    scores : iterable of str
        Subset of ``("max", "tba", "tba_per_window", "trap", "pf")``.
    return_profiles : bool
        Also return per-window TRAP/PF occupancy profiles.
    hard_mask : bool
        Treat lowercase bases as ambiguous.

    Returns
    -------
    pandas.DataFrame
        One row per sequence: ``seq_id``, ``length``, ``n_win`` and the score
        columns. Parameters are stored in ``df.attrs``.
    dict, optional
        When ``return_profiles``: ``{seq_id: {"trap": (2, n), "pf": (2, n)}}``.
    """
    import pandas as pd

    scores = normalise_scores(scores)
    strands = validate_strands(strands)
    records = _as_records(seqs)
    encoded = [encode(s, hard_mask=hard_mask) for _, s in records]
    em = resolve_energy(pwm, lam, pseudocount, background, encoded=encoded)
    r0 = resolve_r0(R0, em)
    validate_r0(r0)
    ln_r0 = float(np.log(r0))

    n = len(records)
    cols = {c: np.full(n, np.nan) for c in score_columns(scores)}
    E_list = [window_energies(s, em, strands=strands) for s in encoded]
    profiles: dict = {}
    for i, E_win in enumerate(E_list):
        L = encoded[i].shape[0]
        if "max" in scores:
            cols["max_score"][i] = max_site(E_win, em).value
        if "tba" in scores:
            cols["ln_tba"][i] = tba(E_win).value
        if "tba_per_window" in scores:
            cols["tba_per_window"][i] = tba_per_window(E_win).value
        if "trap" in scores:
            res = trap(E_win, em, r0)
            cols["trap"][i] = res.value
            cols["trap_per_bp"][i] = res.value / L if L else np.nan
            if return_profiles:
                profiles.setdefault(records[i][0], {})["trap"] = res.profile
    if "pf" in scores:
        pf_vals, pf_occ = _pf_many(E_list, em.width, ln_r0)
        cols["partition_fn"][:] = pf_vals
        if return_profiles:
            for i, occ in enumerate(pf_occ):
                profiles.setdefault(records[i][0], {})["pf"] = split_strands(occ, E_list[i])

    df = pd.DataFrame(
        {
            "seq_id": [r[0] for r in records],
            "length": [e.shape[0] for e in encoded],
            "n_win": [E.shape[1] for E in E_list],
            **cols,
        }
    )
    df.attrs.update(run_params(em, r0, strands))
    if return_profiles:
        return df, profiles
    return df


def _pf_many(E_list: list[np.ndarray], w: int, ln_r0: float):
    """Partition-function score for many sequences, batched by similar length."""
    n = len(E_list)
    vals = np.zeros(n)
    occs: list[np.ndarray | None] = [None] * n
    lengths = np.array([E.shape[1] for E in E_list])
    order = np.argsort(lengths, kind="stable")
    start = 0
    while start < n:
        # Batch sequences whose lengths are within 25% of each other to limit padding.
        stop = start + 1
        while stop < n and stop - start < _PF_BATCH and lengths[order[stop]] <= 1.25 * max(lengths[order[start]], 1):
            stop += 1
        idx = order[start:stop]
        m = int(lengths[idx].max())
        ln_a = np.full((len(idx), m), -np.inf)
        for r, i in enumerate(idx):
            if lengths[i]:
                ln_a[r, : lengths[i]] = ln_activity(E_list[i], ln_r0)
        occ, _ = pf_occupancy_batch(ln_a, w)
        for r, i in enumerate(idx):
            occs[i] = occ[r, : lengths[i]]
            vals[i] = occs[i].sum()
        start = stop
    return vals, occs


def run_params(em: EnergyMatrix, r0: float, strands: str, **extra) -> dict:
    """Parameters echoed into output headers."""
    p = em.pseudocount
    pc = f"{p[0]:.6g}" if np.allclose(p, p[0]) else "auto(sqrt N): " + ",".join(f"{x:.4g}" for x in p)
    d = {
        "motif": em.name,
        "width": em.width,
        "lambda": em.lam,
        "pseudocount": pc,
        "background": ",".join(f"{b}:{x:.6f}" for b, x in zip("ACGT", em.background)),
        "R0": r0,
        "strands": strands,
    }
    d.update(extra)
    return d
