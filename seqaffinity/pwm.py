"""PWM parsers (JASPAR, MEME, bare count matrices) and conversion to energies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .params import DEFAULT_LAMBDA, DEFAULT_PROBABILITY_DEPTH, validate_lambda

ALPHABET = "ACGT"


@dataclass(frozen=True)
class PWM:
    """A position weight matrix stored as counts.

    Attributes
    ----------
    counts : np.ndarray
        ``(w, 4)`` float array of counts, columns in ``ACGT`` order. Probability
        matrices are stored as pseudo-counts at ``depth``.
    name : str
        Motif identifier.
    is_probability : bool
        True if the source matrix held probabilities rather than counts.
    depth : float or None
        Assumed number of sites for probability matrices.
    """

    counts: np.ndarray
    name: str = "motif"
    is_probability: bool = False
    depth: float | None = None

    def __post_init__(self):
        c = np.asarray(self.counts, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 4:
            raise ValueError(f"PWM counts must have shape (w, 4), got {c.shape}")
        if c.shape[0] == 0:
            raise ValueError("PWM has zero width")
        if not np.all(np.isfinite(c)) or np.any(c < 0):
            raise ValueError("PWM counts must be finite and non-negative")
        if np.any(c.sum(axis=1) <= 0):
            raise ValueError("every PWM column must have a positive total")
        c.setflags(write=False)
        object.__setattr__(self, "counts", c)

    @property
    def width(self) -> int:
        return self.counts.shape[0]

    @classmethod
    def from_matrix(
        cls,
        matrix,
        name: str = "motif",
        depth: float | None = None,
    ) -> "PWM":
        """Build a PWM from a ``(w, 4)`` matrix of counts or probabilities.

        A matrix whose rows all sum to 1 (within 1e-3) is treated as probabilities
        and converted to pseudo-counts at ``depth`` (default 100).
        """
        m = np.asarray(matrix, dtype=np.float64)
        if m.ndim != 2 or m.shape[1] != 4:
            raise ValueError(f"matrix must have shape (w, 4), got {m.shape}")
        if m.size and np.allclose(m.sum(axis=1), 1.0, atol=1e-3):
            d = float(depth) if depth else DEFAULT_PROBABILITY_DEPTH
            m = m / m.sum(axis=1, keepdims=True)
            return cls(m * d, name=name, is_probability=True, depth=d)
        return cls(m, name=name)


@dataclass(frozen=True)
class EnergyMatrix:
    """Mismatch energies (kT) derived from a PWM.

    Attributes
    ----------
    E : np.ndarray
        ``(w, 4)`` energies; the best base at each position has energy 0.
    E_bg : np.ndarray
        ``(w,)`` background-weighted mean energy, used for ambiguity codes.
    lam : float
        Scaling constant λ.
    pseudocount : np.ndarray
        ``(w,)`` pseudocount actually used per column.
    background : np.ndarray
        ``(4,)`` background composition.
    lo_max_sum : float
        ``Σ_j max_b LO[j, b]``; log-odds window score is ``lo_max_sum − λ·E_win``.
    name : str
        Motif name.
    """

    E: np.ndarray
    E_bg: np.ndarray
    lam: float
    pseudocount: np.ndarray
    background: np.ndarray
    lo_max_sum: float
    name: str = "motif"
    E_ext: np.ndarray = field(init=False, repr=False)
    Ec_ext: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Column 4 holds the ambiguity energy so that index -1 → 4 is a plain gather.
        ext = np.concatenate([self.E, self.E_bg[:, None]], axis=1)
        # Complemented alphabet (A<->T, C<->G) for the minus strand.
        cext = np.concatenate([self.E[:, ::-1], self.E_bg[:, None]], axis=1)
        object.__setattr__(self, "E_ext", np.ascontiguousarray(ext))
        object.__setattr__(self, "Ec_ext", np.ascontiguousarray(cext))

    @property
    def width(self) -> int:
        return self.E.shape[0]


def resolve_background(background) -> np.ndarray:
    """Return a normalised ``(4,)`` background from 'uniform', a sequence or a string."""
    if background is None or (isinstance(background, str) and background == "uniform"):
        return np.full(4, 0.25)
    if isinstance(background, str):
        parts = [float(x) for x in background.split(",")]
        background = parts
    q = np.asarray(background, dtype=np.float64)
    if q.shape != (4,) or np.any(q <= 0) or not np.all(np.isfinite(q)):
        raise ValueError("background must be four positive numbers (A, C, G, T)")
    return q / q.sum()


def to_energy(
    pwm: PWM,
    lam: float = DEFAULT_LAMBDA,
    pseudocount: float | str = "auto",
    background="uniform",
) -> EnergyMatrix:
    """Convert a PWM to a mismatch-energy matrix.

    ``LO[j,b] = ln((n[j,b] + p·q[b]) / (N_j + p)) − ln q[b]`` and
    ``E[j,b] = (max_b' LO[j,b'] − LO[j,b]) / λ``.

    Parameters
    ----------
    pwm : PWM
    lam : float
        Scaling constant λ (> 0). Smaller λ gives a steeper landscape.
    pseudocount : float or "auto"
        Total pseudocount per column, distributed by background. ``"auto"`` uses
        ``√N_j`` (N_j = column total, or the depth for probability matrices).
        ``p = 4`` with a uniform background reproduces TRAP's +1-per-cell rule.
    background : "uniform", str "a,c,g,t" or array-like of 4
        Background composition.

    Returns
    -------
    EnergyMatrix
    """
    lam = validate_lambda(lam)
    q = resolve_background(background)
    n = pwm.counts
    N = n.sum(axis=1)
    if isinstance(pseudocount, str):
        if pseudocount != "auto":
            raise ValueError(f"pseudocount must be a number or 'auto', got {pseudocount!r}")
        p = np.sqrt(N)
    else:
        p = np.full(n.shape[0], float(pseudocount))
    if np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("pseudocount must be non-negative")
    num = n + p[:, None] * q[None, :]
    if np.any(num <= 0):
        raise ValueError("zero counts with zero pseudocount give infinite energies; use pseudocount > 0")
    lo = np.log(num / (N + p)[:, None]) - np.log(q)[None, :]
    lo_max = lo.max(axis=1)
    E = (lo_max[:, None] - lo) / lam
    # Exact zero at the consensus (subtraction of a value from itself is already 0,
    # but ties and -0.0 are normalised here).
    E[E < 0] = 0.0
    E = E + 0.0
    E_bg = E @ q
    return EnergyMatrix(
        E=E,
        E_bg=E_bg,
        lam=lam,
        pseudocount=p,
        background=q,
        lo_max_sum=float(lo_max.sum()),
        name=pwm.name,
    )


# --------------------------------------------------------------------------- parsers

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _check_alphabet(letters) -> None:
    if sorted(letters) != sorted(ALPHABET):
        raise ValueError(f"unsupported alphabet {''.join(letters)!r}; only ACGT is supported")


def parse_jaspar(text: str, name: str | None = None) -> PWM:
    """Parse a JASPAR matrix (bracketed ``A [ ... ]`` rows or four bare rows).

    If the file holds several motifs, the first is returned unless ``name``
    matches an ID or name in a header line.
    """
    blocks: list[tuple[str, list[str]]] = []
    header, rows = None, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if rows:
                blocks.append((header or "motif", rows))
            header, rows = line[1:].strip(), []
        else:
            rows.append(line)
    if rows:
        blocks.append((header or "motif", rows))
    if not blocks:
        raise ValueError("no JASPAR matrix found")

    chosen = blocks[0]
    if name is not None:
        hits = [b for b in blocks if name in b[0].split()]
        if not hits:
            raise ValueError(f"motif {name!r} not found")
        chosen = hits[0]
    header, rows = chosen
    if len(rows) != 4:
        raise ValueError(f"JASPAR matrix must have 4 rows, got {len(rows)}")

    labelled = {}
    bare = []
    for row in rows:
        m = re.match(r"^([A-Za-z])\s*\[(.*)\]\s*$", row) or re.match(r"^([A-Za-z])\s*[:|]?\s+(.*)$", row)
        if m:
            labelled[m.group(1).upper()] = [float(x) for x in re.findall(_NUM, m.group(2))]
        else:
            bare.append([float(x) for x in re.findall(_NUM, row)])
    if labelled and bare:
        raise ValueError("JASPAR rows mix labelled and unlabelled formats")
    if labelled:
        _check_alphabet(list(labelled))
        mat = [labelled[b] for b in ALPHABET]
    else:
        mat = bare
    widths = {len(r) for r in mat}
    if len(widths) != 1:
        raise ValueError("JASPAR rows have different lengths")
    motif_name = header.split()[0] if header else "motif"
    return PWM.from_matrix(np.array(mat).T, name=motif_name)


def parse_meme(text: str, name: str | None = None) -> PWM:
    """Parse the first (or the named) ``letter-probability matrix`` in a MEME file."""
    alph = re.search(r"^ALPHABET=\s*(\S+)", text, re.M)
    if alph:
        _check_alphabet(list(alph.group(1).upper()))
    lines = text.splitlines()
    motif_name = "motif"
    i = 0
    found = None
    current: tuple = ()
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("MOTIF"):
            motif_name = line.split()[1] if len(line.split()) > 1 else "motif"
            alt = line.split()[2] if len(line.split()) > 2 else None
            current = (motif_name, alt)
        if line.startswith("letter-probability matrix"):
            if name is None or name in current:
                attrs = dict(re.findall(r"(\w+)=\s*(" + _NUM + ")", line))
                w = int(float(attrs["w"])) if "w" in attrs else None
                nsites = float(attrs["nsites"]) if "nsites" in attrs else None
                if "alength" in attrs and int(float(attrs["alength"])) != 4:
                    raise ValueError("only alength=4 (ACGT) MEME motifs are supported")
                rows = []
                j = i + 1
                while j < len(lines) and (w is None or len(rows) < w):
                    vals = re.findall(_NUM, lines[j])
                    if len(vals) == 4 and not lines[j].strip().startswith(("URL", "MOTIF")):
                        rows.append([float(v) for v in vals])
                    elif rows:
                        break
                    j += 1
                found = (motif_name, np.array(rows), nsites)
                break
        i += 1
    if found is None:
        raise ValueError("no letter-probability matrix found" + (f" for {name!r}" if name else ""))
    motif_name, mat, nsites = found
    if mat.ndim != 2 or mat.shape[0] == 0:
        raise ValueError("empty MEME matrix")
    if np.allclose(mat.sum(axis=1), 1.0, atol=1e-2):
        d = nsites if nsites and nsites > 0 else DEFAULT_PROBABILITY_DEPTH
        mat = mat / mat.sum(axis=1, keepdims=True)
        return PWM(mat * d, name=motif_name, is_probability=True, depth=d)
    return PWM.from_matrix(mat, name=motif_name)


def parse_counts(text: str, name: str = "motif") -> PWM:
    """Parse a bare text matrix: ``w`` rows × 4 columns (ACGT), counts or probabilities.

    A matrix with 4 rows and a different number of columns is transposed. Lines
    starting with ``#`` and a single optional ``A C G T`` header line are ignored.
    """
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.replace(",", " ").split()
        if all(re.fullmatch(r"[A-Za-z]", t) for t in toks):
            _check_alphabet([t.upper() for t in toks])
            if [t.upper() for t in toks] != list(ALPHABET):
                raise ValueError("column header must be in ACGT order")
            continue
        try:
            rows.append([float(t) for t in toks])
        except ValueError as exc:
            raise ValueError(f"cannot parse matrix line {raw!r}") from exc
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError("matrix rows have different lengths")
    mat = np.array(rows)
    if mat.shape[1] != 4 and mat.shape[0] == 4:
        mat = mat.T
    if mat.shape[1] != 4:
        raise ValueError(f"matrix must have 4 columns (ACGT), got shape {mat.shape}")
    return PWM.from_matrix(mat, name=name)


def load_pwm(path, fmt: str | None = None, name: str | None = None) -> PWM:
    """Load a PWM from a file.

    Parameters
    ----------
    path : str or Path
    fmt : {"jaspar", "meme", "counts"} or None
        Detected from content/extension when None.
    name : str or None
        Motif to select from a multi-motif file.
    """
    path = Path(path)
    text = path.read_text()
    if fmt is None:
        if "letter-probability matrix" in text:
            fmt = "meme"
        elif text.lstrip().startswith(">") or path.suffix.lower() in (".jaspar", ".pfm"):
            fmt = "jaspar"
        else:
            fmt = "counts"
    if fmt == "jaspar":
        return parse_jaspar(text, name=name)
    if fmt == "meme":
        return parse_meme(text, name=name)
    if fmt == "counts":
        return parse_counts(text, name=name or path.stem)
    raise ValueError(f"unknown PWM format {fmt!r}")
