"""Small statistics shared by the summarize and fit stages.

`wilson` is the interval every rate table in the paper reports (90% two-sided by default,
z = 1.645). `auroc_ranks` is the exact Mann-Whitney AUROC on numpy arrays with ties counted
half; `directions.auroc` is the same quantity on torch tensors.

Pure stdlib + numpy, importable from the eval env.
"""

import math

Z90 = 1.6448536269514722   # two-sided 90%
Z95 = 1.959963984540054    # two-sided 95%


def wilson(k, n, z=Z90):
    """Wilson score interval for k successes in n trials, clamped to [0, 1].

    Returns (lo, hi); (nan, nan) when n == 0. z = 1.645 gives the paper's 90% interval.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def auroc_ranks(pos, neg):
    """Exact AUROC of two 1-D score arrays via midranks (ties count half); nan if either is empty."""
    import numpy as np

    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    sv = allv[order]
    r = np.empty(len(allv))
    i = 0
    while i < len(sv):   # midrank of each tie group
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rs = r[:len(pos)].sum()
    return float((rs - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
