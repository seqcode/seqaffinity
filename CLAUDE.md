# seqaffinity — conventions

- The "Method" section of README.md is the spec for the scoring conventions. Do not change a convention without updating it there.
- Core modules (pwm, encode, windows, pf, scores, streams, aggregate) depend on numpy only; numba is an optional import with a NumPy fallback. pandas is allowed in api.py, cli.py and writers.py; pyfaidx only in genome.py; pyBigWig only in writers.py.
- Alphabet order is ACGT everywhere. Ambiguity codes encode as -1.
- Energies are in kT, consensus = 0, mismatches positive. Affinity K = exp(-E) ≤ 1. Smaller λ = steeper landscape.
- Both strands are stored at forward start coordinates: E_win[1, i] is the minus-strand energy of the w bases starting at forward position i.
- Partition-function recursions run in log space with prefix-length indexing (Z_f[x] = bases [0, x)). Sums of K and occupancy run in linear float64.
- Window sums are direct segment sums (np.add.reduceat), never prefix-sum differences.
- Genomic coordinates are 0-based, half-open. A motif window counts toward a genomic window only if it is fully contained in it.
- Every scoring and stream function is pure: numpy in, numpy/ScoreResult out. No file I/O below api.py / scan.py.
- Scan output is streamed; never materialise all windows of a genome in memory.
- Float comparisons in tests use explicit tolerances, except where bit-identity is part of the spec (chunk-size invariance of segment sums, max-site).
- Every change keeps `python -m pytest -q` green. User-visible changes get a note in CHANGELOG.md.
- Do not add thresholds, p-values or site calling. This package scores whole sequences and windows.
- Type hints and docstrings on all public functions; numpy docstring style.
- Install for development with `pip install -e '.[scan,bigwig,dev]'`. Run tests with `python -m pytest -q`, and performance tests with `-m slow`. bigWig tests skip without pyBigWig. The MatrixRider/tRap fixtures come from `scripts/make_reference_fixtures.R`, with R packages in a project-local `.Rlib/`.
