import numpy as np


def eval_poly(gf, coeffs, x):
    acc = gf.dtype(0)
    for c in reversed(coeffs):
        acc = gf.mul_table[acc, x] ^ c
    return acc


def shamir_share(gf, value, n_parties, t, points, rng):
    value = np.asarray(value, dtype=gf.dtype)
    flat = value.reshape(-1)
    coeffs = [flat] + [gf.rand(rng, flat.shape) for _ in range(t)]
    shares = np.zeros((n_parties, flat.size), dtype=gf.dtype)
    for i in range(n_parties):
        acc = np.zeros(flat.shape, dtype=gf.dtype)
        for c in reversed(coeffs):
            acc = gf.mul(acc, points[i]) ^ c
        shares[i] = acc
    return shares.reshape((n_parties,) + value.shape)


def lagrange_weights(gf, points, subset):
    w = []
    for i in subset:
        num = gf.dtype(1)
        den = gf.dtype(1)
        for j in subset:
            if i == j:
                continue
            num = gf.mul_table[num, points[j]]
            den = gf.mul_table[den, points[i] ^ points[j]]
        w.append(gf.mul_table[num, gf.inv_table[den]])
    return w


def reconstruct(gf, points, shares, subset):
    w = lagrange_weights(gf, points, subset)
    acc = np.zeros(shares.shape[1:], dtype=gf.dtype)
    for k, i in enumerate(subset):
        acc ^= gf.mul(shares[i], w[k])
    return acc


def poly_divmod(gf, num, den):
    num = list(num)
    dd = len(den) - 1
    while dd >= 0 and den[dd] == 0:
        dd -= 1
    if dd < 0:
        return None, None
    q = [gf.dtype(0)] * max(1, len(num) - dd)
    inv = gf.inv_table[den[dd]]
    for i in range(len(num) - 1, dd - 1, -1):
        c = gf.mul_table[num[i], inv]
        q[i - dd] = c
        if c:
            for j in range(dd + 1):
                num[i - dd + j] ^= gf.mul_table[c, den[j]]
    return q, num[:dd]


def bw_decode(gf, points, ys, t, e):
    n = len(points)
    cols = (t + e + 1) + e
    M = np.zeros((n, cols), dtype=gf.dtype)
    b = np.zeros(n, dtype=gf.dtype)
    for i in range(n):
        x = points[i]
        pw = gf.dtype(1)
        for j in range(t + e + 1):
            M[i, j] = pw
            pw = gf.mul_table[pw, x]
        pw = gf.dtype(1)
        for j in range(e):
            M[i, t + e + 1 + j] = gf.mul_table[ys[i], pw]
            pw = gf.mul_table[pw, x]
        b[i] = gf.mul_table[ys[i], pw]
    sol = gf.solve(M, b)
    if sol is None:
        return None, None
    Q = [gf.dtype(v) for v in sol[: t + e + 1]]
    E = [gf.dtype(v) for v in sol[t + e + 1:]] + [gf.dtype(1)]
    q, r = poly_divmod(gf, Q, E)
    if q is None or any(int(v) != 0 for v in r):
        return None, None
    bad = [i for i in range(n) if int(eval_poly(gf, E, points[i])) == 0]
    return q[: t + 1], bad


def robust_open(gf, points, shares, t, probe=2):
    n = len(points)
    flat = shares.reshape(n, -1)
    ncoord = flat.shape[1]
    maxe = (n - t - 1) // 2
    for e in range(maxe + 1):
        bad = None
        okall = True
        for c in range(min(probe, ncoord)):
            res, b = bw_decode(gf, points, [gf.dtype(flat[i, c]) for i in range(n)], t, e)
            if res is None:
                okall = False
                break
            bad = b if bad is None else sorted(set(bad) | set(b))
        if okall:
            good = [i for i in range(n) if i not in (bad or [])]
            if len(good) < t + 1:
                continue
            val = reconstruct(gf, points, shares, good[: t + 1])
            return val, sorted(bad or [])
    return None, None
