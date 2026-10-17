"""Command-line interface: ``seqaffinity score`` and ``seqaffinity scan``."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .params import DEFAULT_CHUNK_SIZE, DEFAULT_LAMBDA, DEFAULT_MIN_VALID_FRACTION, SCORE_KEYS


def _pseudocount(v: str):
    return v if v == "auto" else float(v)


def _common(p: argparse.ArgumentParser, default_bg: str) -> None:
    p.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA, help="mismatch scaling λ (default 0.7)")
    p.add_argument("--pseudocount", type=_pseudocount, default="auto", help="per-column pseudocount or 'auto' (√N)")
    p.add_argument(
        "--background",
        default=default_bg,
        help=f"uniform, from-seqs, from-genome, or 'a,c,g,t' frequencies (default {default_bg})",
    )
    p.add_argument("--R0", type=float, default=None, help="effective concentration (default: TRAP width formula)")
    p.add_argument("--strands", choices=("both", "plus"), default="both")
    p.add_argument("--scores", default=",".join(SCORE_KEYS), help="comma-separated subset of " + ",".join(SCORE_KEYS))
    p.add_argument("--hard-mask", action="store_true", help="treat lowercase bases as N")
    p.add_argument("--motif", default=None, help="motif ID/name to select from a multi-motif file")


def _load(args):
    from .pwm import load_pwm

    return load_pwm(args.pwm, name=args.motif)


def cmd_score(args) -> int:
    from .api import score_sequences

    res = score_sequences(
        _load(args),
        args.seqs,
        lam=args.lam,
        pseudocount=args.pseudocount,
        background=args.background,
        R0=args.R0,
        strands=args.strands,
        scores=args.scores,
        return_profiles=bool(args.profile),
        hard_mask=args.hard_mask,
    )
    df, prof = res if args.profile else (res, {})
    out = sys.stdout if args.out == "-" else open(args.out, "w")
    try:
        for k, v in df.attrs.items():
            out.write(f"# {k}={v}\n")
        df.to_csv(out, sep="\t", index=False, float_format="%.10g", na_rep="NA")
    finally:
        if out is not sys.stdout:
            out.close()
    if args.profile:
        if args.profile not in prof:
            raise SystemExit(f"--profile: sequence {args.profile!r} not found")
        _write_profile(args, prof[args.profile])
    return 0


def _write_profile(args, prof: dict) -> None:
    from .api import resolve_energy, _as_records
    from .encode import encode
    from .windows import window_energies

    seqs = dict(_as_records(args.seqs))
    path = args.profile_out or f"{args.profile}.profile.tsv"
    s = encode(seqs[args.profile], hard_mask=args.hard_mask)
    # Recompute energies with the same parameters for the K column.
    encoded = [encode(v, hard_mask=args.hard_mask) for v in seqs.values()]
    em = resolve_energy(_load(args), args.lam, args.pseudocount, args.background, encoded=encoded)
    E = window_energies(s, em, strands=args.strands)
    with open(path, "w") as fh:
        fh.write("start\tstrand\tenergy\tK\ttrap_occ\tpf_occ\n")
        for strand in range(2):
            if args.strands == "plus" and strand == 1:
                continue
            for i in range(E.shape[1]):
                t = prof.get("trap", np.full((2, E.shape[1]), np.nan))[strand, i]
                f = prof.get("pf", np.full((2, E.shape[1]), np.nan))[strand, i]
                fh.write(f"{i}\t{'+-'[strand]}\t{E[strand, i]:.10g}\t{np.exp(-E[strand, i]):.10g}\t{t:.10g}\t{f:.10g}\n")
    print(f"profile written to {path}", file=sys.stderr)


def cmd_scan(args) -> int:
    from .scan import scan_genome

    summary = scan_genome(
        _load(args),
        args.genome,
        W=args.window,
        s=args.step,
        out=args.out,
        fmt=args.format,
        workers=args.workers,
        lam=args.lam,
        pseudocount=args.pseudocount,
        background=args.background,
        R0=args.R0,
        strands=args.strands,
        scores=args.scores,
        pf_mode=args.pf_mode,
        pf_overlap=args.pf_overlap,
        min_valid_fraction=args.min_valid_fraction,
        chunk_size=args.chunk_size,
        keep_tail=args.keep_tail,
        hard_mask=args.hard_mask,
        region=args.region,
    )
    print(
        f"scanned {summary['n_windows']:,} windows in {summary['n_chunks']} chunks "
        f"({summary['fasta_backend']} FASTA backend)",
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="seqaffinity", description="Whole-sequence PWM affinity scoring")
    ap.add_argument("--version", action="version", version=f"seqaffinity {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("score", help="score sequences from a FASTA file")
    sp.add_argument("pwm", help="PWM file (JASPAR, MEME or count matrix)")
    sp.add_argument("seqs", help="FASTA file")
    _common(sp, "uniform")
    sp.add_argument("--out", default="-", help="output TSV (default stdout)")
    sp.add_argument("--profile", metavar="SEQ_ID", help="also write a per-window occupancy table for this sequence")
    sp.add_argument("--profile-out", help="path for --profile (default SEQ_ID.profile.tsv)")
    sp.set_defaults(func=cmd_score)

    gp = sub.add_parser("scan", help="score sliding windows across a genome FASTA")
    gp.add_argument("pwm", help="PWM file (JASPAR, MEME or count matrix)")
    gp.add_argument("genome", help="genome FASTA (plain, gzip or BGZF)")
    _common(gp, "from-genome")
    gp.add_argument("--window", type=int, default=200)
    gp.add_argument("--step", type=int, default=50)
    gp.add_argument("--out", default="-", help="output path; a prefix for bigwig (default stdout)")
    gp.add_argument("--format", choices=("tsv", "bed", "bigwig"), default="tsv")
    gp.add_argument("--workers", type=int, default=1)
    gp.add_argument("--pf-mode", choices=("genome", "isolated"), default="genome")
    gp.add_argument("--pf-isolated", dest="pf_mode", action="store_const", const="isolated", help="alias for --pf-mode isolated")
    gp.add_argument("--pf-overlap", type=int, default=None, help="context flank for genome-mode PF (default 64·w)")
    gp.add_argument("--min-valid-fraction", type=float, default=DEFAULT_MIN_VALID_FRACTION)
    gp.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    gp.add_argument("--keep-tail", action="store_true")
    gp.add_argument("--region", help="chr or chr:start-end (1-based, inclusive)")
    gp.set_defaults(func=cmd_scan)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "score" and args.background == "from-genome":
        raise SystemExit("--background from-genome is only valid for scan; use from-seqs")
    if args.cmd == "scan" and args.background == "from-seqs":
        raise SystemExit("--background from-seqs is only valid for score; use from-genome")
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
