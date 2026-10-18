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

### Inputs (both commands)

| Argument | Description |
| --- | --- |
| `pwm` | The motif file. JASPAR, MEME and plain count-matrix formats are detected from content and extension (see [PWM formats](#pwm-formats)). Counts and probabilities are both accepted. |
| `seqs` (`score`) | A FASTA file of sequences. IUPAC ambiguity codes, `N`, `.` and `-` are scored at the background-mean energy; `U` is read as `T`. Any other character is an error. |
| `genome` (`scan`) | A genome FASTA: plain, gzip or BGZF. Mixed line widths and CRLF line endings are fine. Duplicate sequence names are an error. Uncompressed files use pyfaidx, when it is installed, for fast `--region` access; pyfaidx writes a `.fai` index next to the file. |

### Model options (both commands)

| Option | Default | Description |
| --- | --- | --- |
| `--lambda` | `0.7` | Mismatch scaling λ: energies are `(max LO − LO) / λ`, in kT. Must be > 0. Smaller values give a steeper landscape, where TBA, TRAP and the partition function approach max-site. Larger values flatten it, so weak windows count more. 0.7 is TRAP's calibrated value, and R0 is calibrated together with it, so change both if you change one. |
| `--pseudocount` | `auto` | The total pseudocount per motif column, shared out by the background (`n + p·q`). `auto` uses `√N` per column, where `N` is the column's count total, or the depth for probability matrices. A number uses that value for every column. `0` is allowed only if the matrix has no zero counts. TRAP's own +1 per cell equals `--pseudocount 4` with a uniform background. A larger pseudocount caps the per-position penalty lower (about `ln(N/(p·q))/λ`). |
| `--background` | `uniform` (`score`); `from-genome` (`scan`) | The base composition used in the log-odds and for ambiguity codes. `uniform` is 0.25 each. `a,c,g,t` takes four positive numbers, which are normalised, e.g. `0.3,0.2,0.2,0.3`. `from-seqs` (`score` only) uses the composition of the input sequences, plus one per base. It's circular if those sequences are the set being tested. `from-genome` (`scan` only) counts the scanned sequence (just the region with `--region`) in an extra pass before scoring, plus one per base. Bases excluded by `--hard-mask` aren't counted. The value used is written to the output header; pass it back as `a,c,g,t` to reproduce scan values with `score`. |
| `--R0` | `exp(0.584·w − 5.66)`, capped at 10 | The effective TF concentration for `trap` and `partition_fn`; it doesn't affect the other scores. The default is TRAP's width formula (≈ 1.2 at w = 10). The cap engages from w = 14 and prints a warning. It's a seqaffinity addition, so results there differ from tRap. Must be > 0. |
| `--strands` | `both` | `both` scores both strands. `plus` scores only the given strand, for strand-specific assays. With `plus`, `tba_per_window` divides by `n` rather than `2n`, and results are no longer invariant to reverse-complementing the input. |
| `--scores` | all five | A comma-separated subset of `max`, `tba`, `tba_per_window`, `trap`, `pf`. Output columns follow the selection (see [Output columns](#output-columns)). Leaving out `pf` saves the most time in `scan`. |
| `--hard-mask` | off | Treat lowercase (soft-masked) bases as `N`. Off by default, so soft-masked repeats are scored. |
| `--motif` | first motif | Selects a motif from a multi-motif file. For JASPAR, it matches any word of the `>` header line (ID or name). For MEME, it matches the `MOTIF` ID or alternate name. For a plain count matrix, it only sets the motif name. |

### `score` options

| Option | Default | Description |
| --- | --- | --- |
| `--out` | stdout | The output TSV path; `-` also means stdout. |
| `--profile SEQ_ID` | none | Also write a per-window table for one sequence. Its columns are `start` (0-based within the sequence), `strand`, `energy`, `K`, `trap_occ` and `pf_occ`. Occupancy columns for scores not selected in `--scores` are `nan`. With `--strands plus`, minus-strand rows are left out. |
| `--profile-out` | `SEQ_ID.profile.tsv` | Where to write `--profile`. |

### `scan` options

| Option | Default | Description |
| --- | --- | --- |
| `--window` | `200` | Window size W in bp. Must be ≥ the motif width. A genomic window is scored from the `W − w + 1` motif windows lying entirely inside it. |
| `--step` | `50` | The distance s between window starts; requires `1 ≤ s ≤ W`. Windows are `[k·s, k·s + W)`, aligned to each contig's start. `s = W` gives non-overlapping tiles. Smaller steps give more rows but cost little extra compute. |
| `--out` | stdout | The output path, with `-` for stdout. For `--format bigwig` it's a prefix: a trailing `.bw` is removed, and the scan writes `PREFIX.<column>.bw` per score column plus `PREFIX.params.txt` with the run parameters. |
| `--format` | `tsv` | `tsv` writes `#` parameter lines, a header row, then one row per window. `bed` is the same, except the header row is commented, so the file reads as BED with extra columns. `bigwig` writes one bigWig per score and needs pyBigWig (see below). |
| `--workers` | `1` | The number of worker processes. Output stays in FASTA order with ascending starts, and is identical for any number of workers. At most `2 × workers` chunks are in memory at once. |
| `--pf-mode` | `genome` | `genome` computes each motif window's partition-function occupancy in the context of the surrounding sequence, so steric exclusion continues across window edges. `isolated` reruns the partition function inside each window, which reproduces `score` on the extracted window exactly and costs more. |
| `--pf-isolated` | | Short for `--pf-mode isolated`. |
| `--pf-overlap` | `64·w` | The bases of flanking context, each side of a chunk, used by genome-mode partition function (clipped at contig ends). Only used when `pf` is selected and `--pf-mode genome`. Larger values move chunked results closer to a whole-chromosome calculation; at the default, the tests find a relative difference below 1e-8. |
| `--min-valid-fraction` | `0.5` | A window gets `NA` scores when its N-free motif windows number fewer than this fraction of `W − w + 1`. `n_valid` is always written, so you can re-filter later. `0` never gives NA; all-N windows are then scored at background energy. |
| `--chunk-size` | `10,000,000` | Window starts are grouped into blocks of this many bases. Each block is one unit of work, carrying the sequence its windows need plus any partition-function flank. This controls memory (roughly 1 GB per worker at 10 Mb, an estimate) and how finely work is split. Results are bit-identical across chunk sizes, except genome-mode partition function (within ~1e-8). Must be ≥ `--step`. |
| `--keep-tail` | off | Add one shorter window at each contig end when the last full window stops short of it: `[a_last + s, L)`, or `[0, L)` for a contig shorter than W. The tail is added only if it's at least w long, and it's scored on its own length. Without this option, contigs shorter than W produce no rows. |
| `--region` | whole genome | Scan only `chrom` or `chrom:start-end`. Coordinates are 1-based and inclusive, and commas are allowed, e.g. `chr1:1-10,000,000`. Only windows lying entirely inside the region are written, on the same grid as a full scan, so rows match the full scan's. Genome-mode partition function still uses flanking sequence outside the region. |

### Python API

The CLI options map onto keyword arguments:

```python
sa.score_sequences(pwm, seqs, lam=0.7, pseudocount="auto", background="uniform", R0=None,
                   strands="both", scores=("max", "tba", "tba_per_window", "trap", "pf"),
                   return_profiles=False, hard_mask=False)
sa.scan_genome(pwm, fasta_path, W=200, s=50, out="-", fmt="tsv", workers=1, lam=0.7,
               pseudocount="auto", background="from-genome", R0=None, strands="both",
               scores=..., pf_mode="genome", pf_overlap=None, min_valid_fraction=0.5,
               chunk_size=10_000_000, keep_tail=False, hard_mask=False, region=None,
               fasta_backend="auto", pf_backend="auto")
```

Differences from the CLI:

- **`pwm`** can be a `PWM`, an `EnergyMatrix` (from `sa.to_energy(...)`), a file path, or a `(w, 4)` array in ACGT order. An `EnergyMatrix` is used as is: `lam`, `pseudocount` and `background` are then ignored.
- **`seqs`** can be a FASTA path, a single sequence string, a list of strings (named `seq1`, `seq2`, …), a list of `(id, sequence)` tuples, or a dict.
- **`background`** also accepts a length-4 array.
- **`return_profiles=True`** returns `(df, profiles)`, where `profiles[seq_id]` holds `"trap"` and/or `"pf"` arrays of shape `(2, n)`: plus strand, then minus strand, by forward start position.
- **`score_one(pwm, seq, ...)`** returns a `ScoreResult` per score, with value, profile and units, plus the raw `(2, n)` energy array `E_win`.
- **`scan_genome(out=...)`** also accepts an open text file. It returns a summary dict: window and chunk counts, the FASTA backend used, and the run parameters.
- **`fasta_backend`** (`auto`, `pyfaidx`, `stream`) and **`pf_backend`** (`auto`, `numba`, `numpy`) choose implementations; `auto` is right unless you're testing.
- **`load_pwm(path, fmt=None, name=None)`** takes `fmt` as `"jaspar"`, `"meme"` or `"counts"` to override detection. `name` works like `--motif`.

### Output columns

| Column | Appears in | Description |
| --- | --- | --- |
| `seq_id`, `length`, `n_win` | `score` | Sequence name, length, and number of motif windows per strand (`L − w + 1`, or 0 when shorter than the motif). |
| `chrom`, `start`, `end` | `scan` | Window coordinates, 0-based and half-open. |
| `n_valid` | `scan` | Motif windows in this window that contain no ambiguity code. |
| `max_score` | `--scores max` | Best log-odds window score over both strands, in natural log. FIMO reports bits: divide by ln 2 to compare. `NaN` for sequences shorter than the motif. |
| `ln_tba` | `--scores tba` | `ln Σ K` over both strands. `-inf` when there are no windows. |
| `tba_per_window` | `--scores tba_per_window` | `Σ K / (2n)`: mean affinity per motif window, so sequences of different lengths can be compared. |
| `trap`, `trap_per_bp` | `--scores trap` | TRAP expected occupancy `Σ R0·K/(1 + R0·K)`, and that value divided by the sequence or window length. |
| `partition_fn` | `--scores pf` | Expected number of bound molecules when bound sites may not overlap. Always ≤ `trap`. |

Both commands start their output with `# key=value` lines recording the motif, width, λ, pseudocount, background, R0 and strands. `scan` also records the window, step, partition-function mode and overlap, NA threshold, tail, masking, region and genome path.

### PWM formats

- **JASPAR:** a `>ID name` header followed by four rows, either `A [ … ]` (labelled, any order) or four bare rows of numbers in ACGT order. A multi-motif file is read from its first motif unless `--motif` is given.
- **MEME:** the minimal MEME format. The `letter-probability matrix:` line supplies `w` (rows to read), `alength` (must be 4) and `nsites`. Probabilities are converted to counts at depth `nsites` (default 100), which sets the `auto` pseudocount. `ALPHABET=`, if present, must be ACGT. The file's background frequencies are ignored in favour of `--background`.
- **Count matrix:** plain text with `w` rows of four numbers (ACGT), separated by whitespace or commas. A matrix written as four rows of `w` columns is transposed automatically. An optional `A C G T` header line and `#` comment lines are allowed.

In all three formats, rows that each sum to 1 are read as probabilities. Other values are read as counts. Negative values, or a column that sums to 0, are errors.

### Installing pyBigWig

`--format bigwig` needs the optional `pyBigWig` package (`pip install -e '.[bigwig]'`). It compiles C code, so it needs a working C compiler. On macOS that means the Xcode command-line tools, with the license accepted.

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
