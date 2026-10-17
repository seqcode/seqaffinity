# Changelog

## 0.1.0

First release.

- **Scores.** Five whole-sequence scores from one per-window energy array: max-site log-odds, ln TBA, TBA per window, TRAP occupancy, and a steric-exclusion partition function. The partition function is computed in log space, using numba when available or NumPy vectorised across sequences.
- **Inputs.** PWMs in JASPAR (bracketed or bare rows, multi-motif files), MEME (`nsites` used as depth) and plain count-matrix format. IUPAC ambiguity codes are scored at background energy, and soft-masked bases can optionally be hard-masked.
- **`seqaffinity score` / `score_sequences`.** One row per sequence with the parameters echoed. Optional per-window occupancy profiles.
- **`seqaffinity scan` / `scan_genome`.** Sliding windows across a genome FASTA (plain, gzip or BGZF), with TSV, BED-like or bigWig output. Includes an ordered multi-process pool, `--region`, `--keep-tail`, an NA rule for N-rich windows, and genome-context or per-window (`--pf-isolated`) partition function. Output is bit-identical for any chunk size or worker count.
- **Validation.**
  - A naive reference implementation of all five scores.
  - Brute-force enumeration for the partition function.
  - Scan-versus-score equivalence on a synthetic genome.
  - Agreement with MatrixRider and tRap to 1e-9 once their conventions are applied. MatrixRider replaces zero counts by 1 and counts only the better strand per window. tRap combines strands as `P⁺ + (1 − P⁺)·P⁻`. Both handle N differently.
  - A notebook comparing AUROC across scores and λ.
- **Performance.** 10,000 × 200 bp sequences score in about 0.8 s. A 200 Mb synthetic genome scans in about 30 s (scores 1–4) or 50 s (all five) on 8 workers.
