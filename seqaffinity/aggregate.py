"""Window aggregation by direct segment sums and maxima.

A genomic window starting at stream index ``a`` with ``n`` motif windows covers
stream entries ``[a, a + n)``. Sums use ``np.add.reduceat`` on interleaved
``[start, stop)`` pairs rather than prefix-sum differences: each window's value
then depends only on its own entries (no cancellation, bit-identical across
chunkings).
"""

from __future__ import annotations

import numpy as np

_MAX_KEYS = {"best"}


def segment_reduce(x: np.ndarray, starts: np.ndarray, n: np.ndarray, op=np.add) -> np.ndarray:
    """Reduce ``x[starts[i] : starts[i] + n[i]]`` with ``op`` for every ``i``.

    Parameters
    ----------
    x : np.ndarray
        1-D stream.
    starts, n : np.ndarray
        Segment starts and lengths (every ``n >= 1``, segments inside ``x``).
    op : numpy ufunc
        ``np.add`` or ``np.maximum``.
    """
    starts = np.asarray(starts, dtype=np.intp)
    n = np.asarray(n, dtype=np.intp)
    if starts.size == 0:
        return np.zeros(0, dtype=x.dtype)
    if np.any(n < 1) or starts.min() < 0 or (starts + n).max() > x.shape[0]:
        raise ValueError("segments must be non-empty and lie inside the stream")
    # Sentinel so that stop == len(x) is a valid reduceat index.
    xs = np.concatenate([x, np.zeros(1, dtype=x.dtype)])
    idx = np.empty(2 * starts.size, dtype=np.intp)
    idx[0::2] = starts
    idx[1::2] = starts + n
    return op.reduceat(xs, idx)[0::2]


def aggregate(streams: dict[str, np.ndarray], starts: np.ndarray, n_motif: np.ndarray) -> dict[str, np.ndarray]:
    """Aggregate every stream over the given windows.

    ``best`` is reduced with a maximum; every other stream is summed.

    Parameters
    ----------
    streams : dict[str, np.ndarray]
    starts : np.ndarray
        Stream index of each window's first motif window.
    n_motif : np.ndarray
        Number of motif windows per genomic window (``W − w + 1`` for full windows).

    Returns
    -------
    dict[str, np.ndarray]
        Same keys as ``streams``.
    """
    return {
        key: segment_reduce(x, starts, n_motif, np.maximum if key in _MAX_KEYS else np.add)
        for key, x in streams.items()
    }
