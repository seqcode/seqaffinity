"""Regenerate ten.fa and expected.tsv (expected values come from tests/reference_impl.py).

Run from the repository root: python tests/fixtures/make_fixtures.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from reference_impl import scores  # noqa: E402

ARNT = [[4, 16, 0, 0], [19, 0, 1, 0], [0, 20, 0, 0], [0, 0, 20, 0], [0, 0, 0, 20], [0, 0, 20, 0]]
COLS = ["max_score", "ln_tba", "tba_per_window", "trap", "trap_per_bp", "partition_fn"]

rng = np.random.default_rng(2026)
recs = []
for i in range(10):
    L = int(rng.integers(100, 201))
    s = list(rng.choice(list("ACGT"), size=L))
    for pos in rng.integers(0, L - 6, size=i % 4):
        s[pos : pos + 6] = list("CACGTG")
    if i == 7:
        s[20:30] = list("NNNNNNNNNN")
    if i == 8:
        s[40:60] = [c.lower() for c in s[40:60]]
    recs.append((f"seq{i + 1}", "".join(s)))

with open(HERE / "ten.fa", "w") as fh:
    for name, seq in recs:
        fh.write(f">{name}\n")
        for j in range(0, len(seq), 60):
            fh.write(seq[j : j + 60] + "\n")

with open(HERE / "expected.tsv", "w") as fh:
    fh.write("seq_id\tlength\t" + "\t".join(COLS) + "\n")
    for name, seq in recs:
        r = scores(seq, ARNT, lam=0.7, pseudocount="auto")
        fh.write(f"{name}\t{len(seq)}\t" + "\t".join(f"{r[c]:.15g}" for c in COLS) + "\n")
