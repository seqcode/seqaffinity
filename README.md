# seqaffinity

Whole-sequence PWM affinity scoring. Given a PWM and some sequences (or a genome), `seqaffinity` computes five scores from one per-window energy array:

| Column | Score | References |
| --- | --- | --- |
| `max_score` | best-site log-odds (natural log), the FIMO/HOMER-style baseline | [3], [4] |
| `ln_tba` | log total binding affinity, `ln Σ K` over both strands | [5] |
| `tba_per_window` | mean affinity per motif window | — (length-normalised TBA, added here) |
| `trap`, `trap_per_bp` | TRAP expected occupancy `Σ R0·K/(1+R0·K)`, and per base | [1], [2] |
| `partition_fn` | expected bound molecules with steric exclusion (1-D partition function) | [7], [8], [9] |

Energies are in kT, with the consensus at 0 and `K = exp(−E) ≤ 1`. This is the mismatch-energy model behind TRAP [1, 6]. [Method](#method) gives the exact definitions. [References](#references) says how each score relates to its published source.

## Install

Requires Python ≥ 3.11. From a clone of this repository:

```bash
pip install -e '.[scan,dev]'        # numba + pyfaidx + pytest
pip install -e '.[bigwig]'          # pyBigWig, for bigWig output (compiles C code)
```

The core needs only NumPy and pandas. numba speeds up the partition function on long sequences, and pyfaidx speeds up `--region` on uncompressed genomes. On macOS, pyBigWig needs a working C compiler (Xcode command-line tools, with the license accepted).

## Score sequences

```bash
seqaffinity score tests/fixtures/MA0004.1.jaspar tests/fixtures/ten.fa --lambda 0.7
seqaffinity score motif.meme peaks.fa --background from-seqs --profile peak42
```

```python
import seqaffinity as sa

df = sa.score_sequences("MA0004.1.jaspar", "peaks.fa", lam=0.7)
df, profiles = sa.score_sequences(pwm, ["ACGT...", ...], return_profiles=True)
```

## Scan a genome

```bash
seqaffinity scan MA0004.1.jaspar hg38.fa.gz --window 200 --step 50 --workers 8 --out arnt.tsv
seqaffinity scan MA0004.1.jaspar hg38.fa --region chr1:1-10,000,000 --format bigwig --out arnt
```

- Windows are `[k·s, k·s + W)`. A motif window counts toward a genomic window only if it lies entirely inside it, so each row equals `score_sequences` on the extracted window.
- Windows where fewer than `--min-valid-fraction` (default 0.5) of the motif windows are free of N are written as `NA`. `n_valid` is always written.
- The background defaults to the composition of the scanned sequence, which is recorded in the `#` header. Pass the same `--background` to `score` to reproduce scan values.
- The partition function defaults to genome-context occupancy (`--pf-mode genome`). `--pf-isolated` instead runs it inside each window.
- Accepted FASTA inputs are plain, gzip and BGZF. Uncompressed files use pyfaidx for fast `--region` access when it is installed.
- bigWig output writes one file per score (`{out}.{column}.bw`). Each window's value goes on its centred `s`-wide step, because bigWig cannot store overlapping intervals.

Performance: a 200 Mb synthetic genome with 8 workers runs in about 30 s (scores 1–4) and about 50 s (all five). Extrapolated to the human genome that is roughly 8 and 13 min.

## Parameters

| Option | Default | Notes |
| --- | --- | --- |
| `--lambda` | 0.7 | Smaller values give a steeper landscape (scores approach max-site); larger values flatten it. |
| `--R0` | `exp(0.584·w − 5.66)`, capped at 10 | TRAP's calibration, fitted at λ = 0.7 under TRAP's +1 pseudocount (equivalent to `--pseudocount 4` with a uniform background). |
| `--pseudocount` | `auto` = √N per column | Probability matrices are converted to counts at depth `nsites`, or 100 if absent. |
| `--strands` | both | `plus` for strand-specific assays. |

## Method

**Energy matrix.** For counts `n[j,b]` (column total `N_j`), pseudocount `p` (spread by the background `q`) and λ:

```
LO[j,b] = ln((n[j,b] + p·q[b]) / (N_j + p)) − ln q[b]
E[j,b]  = (max_b' LO[j,b'] − LO[j,b]) / λ
```

So the best base at each position has energy 0. Ambiguity codes (N and other IUPAC codes) get the background-weighted mean energy `Σ_b q[b]·E[j,b]`.

**Window energies.** For a sequence of length L and motif width w, there are `n = L − w + 1` windows per strand. The minus-strand energy is stored at the forward start of the bases it covers: `E⁻[i] = Σ_j E[j, comp(s[i+w−1−j])]`. Then `K = exp(−E)`, and the log-odds window score is `Σ_j max_b LO[j,b] − λ·E`.

**Scores.** All five sums run over both strands and every window.

- `max_score`: the largest log-odds window score.
- `ln_tba`: `ln Σ K`.
- `tba_per_window`: `Σ K / (2n)`.
- `trap`: `Σ R0·K / (1 + R0·K)`.
- `partition_fn`: the expected number of bound molecules when bound sites may not overlap. Both strands at one start are alternative placements of the same footprint. With `a_i = R0·(K⁺_i + K⁻_i)`, and `Z_f[x]` / `Z_b[y]` the partition functions of bases `[0, x)` and `[y, L)`:

  ```
  Z_f[x] = Z_f[x−1] + a_{x−w}·Z_f[x−w]   (x ≥ w),     Z_f[x] = 1 otherwise
  Z_b[y] = Z_b[y+1] + a_y·Z_b[y+w]       (y ≤ L−w),   Z_b[y] = 1 otherwise
  occupancy_i = a_i·Z_f[i]·Z_b[i+w] / Z_f[L]
  ```

  The recursion is computed in log space. At low R0 it reduces to TRAP; at high R0 it saturates near `L/w`.

**Genome scan.**

- Each genomic window is scored from the motif windows lying entirely inside it.
- Window totals are summed directly over their own motif windows (`np.add.reduceat`), not by differencing running totals. Weak windows can have TBA of 1e-20, which would be lost to cancellation. Summing directly also makes the output bit-identical for any chunk size or worker count.
- Genome-context partition-function occupancies use `64·w` bases of flanking context around each chunk (`--pf-overlap`).

## Validation

```bash
python -m pytest -q            # fast suite
python -m pytest -q -m slow    # performance targets
```

- `tests/reference_impl.py` is a deliberately naive implementation that generates `tests/fixtures/expected.tsv`.
- The partition function is checked against brute-force enumeration.
- Every scan window is checked against `score_sequences` on a 2 Mb synthetic genome.
- `tests/test_reference.py` agrees with MatrixRider and tRap to 1e-9, once each tool's own pseudocount and strand conventions are applied. Its docstring lists those conventions. The fixtures come from `scripts/make_reference_fixtures.R`, which uses R packages in a project-local `.Rlib/`.
- `notebooks/compare_scores.ipynb` compares AUROC per score and per λ. It uses a synthetic benchmark by default; set `GENOME`/`PEAKS_BED` to use real ChIP-seq peaks.

## License

MIT. See [LICENSE](LICENSE).

## References

Where seqaffinity departs from a published method, the entry says how. [tests/test_reference.py](tests/test_reference.py) checks the numbers against MatrixRider [5] and tRap [2].

1. Roider HG, Kanhere A, Manke T, Vingron M. Predicting transcription factor affinities to DNA from a biophysical model. *Bioinformatics* 2007;23(2):134–141. [PubMed 17098775](https://pubmed.ncbi.nlm.nih.gov/17098775/)
   The TRAP model and the source of the default λ = 0.7 and the width formula `ln R0 = 0.584·w − 5.66`. That calibration assumes TRAP's +1 pseudocount; the R0 cap at 10 is a seqaffinity addition.
2. Thomas-Chollier M, Hufton A, Heinig M, O'Keeffe S, El Masri N, Roider HG, Manke T, Vingron M. Transcription factor binding predictions using TRAP for the analysis of ChIP-seq data and regulatory SNPs. *Nature Protocols* 2011;6(12):1860–1869. [doi:10.1038/nprot.2011.409](https://doi.org/10.1038/nprot.2011.409)
   TRAP protocol and the tRap R package. tRap combines strands as `P⁺ + (1 − P⁺)·P⁻`, whereas seqaffinity sums `P⁺ + P⁻`.
3. Grant CE, Bailey TL, Noble WS. FIMO: scanning for occurrences of a given motif. *Bioinformatics* 2011;27(7):1017–1018. [doi:10.1093/bioinformatics/btr064](https://doi.org/10.1093/bioinformatics/btr064)
   Max-site baseline. FIMO reports log-odds in bits, while `max_score` is in nats.
4. Heinz S, Benner C, Spann N, Bertolino E, Lin YC, Laslo P, Cheng JX, Murre C, Singh H, Glass CK. Simple combinations of lineage-determining transcription factors prime cis-regulatory elements required for macrophage and B cell identities. *Molecular Cell* 2010;38(4):576–589. [doi:10.1016/j.molcel.2010.05.004](https://doi.org/10.1016/j.molcel.2010.05.004)
   HOMER, the other max-site reference point.
5. Grassi E, Zapparoli E, Molineris I, Provero P. Total binding affinity profiles of regulatory regions predict transcription factor binding and gene expression in human cells. *PLoS ONE* 2015;10(11):e0143627. [doi:10.1371/journal.pone.0143627](https://doi.org/10.1371/journal.pone.0143627)
   Total binding affinity, implemented in MatrixRider. MatrixRider counts only the better strand per window and replaces zero counts by 1; `ln_tba` sums both strands and uses an additive pseudocount.
6. Berg OG, von Hippel PH. Selection of DNA binding sites by regulatory proteins. Statistical-mechanical theory and application to operators and promoters. *Journal of Molecular Biology* 1987;193(4):723–750. [PubMed 3612791](https://pubmed.ncbi.nlm.nih.gov/3612791/)
   The link between PWM log-odds and binding energy that underlies the energy matrix.
7. Granek JA, Clarke ND. Explicit equilibrium modeling of transcription-factor binding and gene regulation. *Genome Biology* 2005;6(10):R87. [PubMed 16207358](https://pubmed.ncbi.nlm.nih.gov/16207358/)
   GOMER, an equilibrium occupancy model over whole sequences. The planned hook for multi-TF competition follows this model.
8. Wasson T, Hartemink AJ. An ensemble model of competitive multi-factor binding of the genome. *Genome Research* 2009;19(11):2101–2112. [doi:10.1101/gr.093450.109](https://doi.org/10.1101/gr.093450.109)
   COMPETE, an ensemble (partition-function) model of factors that cannot overlap along the genome. `partition_fn` is the single-factor case of this kind of model.
9. Teif VB, Rippe K. Statistical–mechanical lattice models for protein–DNA binding in chromatin. *Journal of Physics: Condensed Matter* 2010;22(41):414105. [doi:10.1088/0953-8984/22/41/414105](https://doi.org/10.1088/0953-8984/22/41/414105)
   A review of transfer-matrix and recursive methods for 1-D lattice binding with steric exclusion. It is the general form of the forward/backward recursion in `seqaffinity/pf.py`.
10. Ambrosini G, Vorontsov I, Penzar D, Groux R, Fornes O, Nikolaeva DD, Ballester B, Grau J, Grosse I, Makeev V, Kulakovskiy I, Bucher P. Insights gained from a comprehensive all-against-all transcription factor binding motif benchmarking study. *Genome Biology* 2020;21(1):114. [doi:10.1186/s13059-020-01996-3](https://doi.org/10.1186/s13059-020-01996-3)
    PWMEval. The validation notebook follows its ChIP-seq design: summit ±100 bp against controls 300 bp downstream.
