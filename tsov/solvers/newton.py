import math
import time

import numpy as np

from ..abb import Shared
from ..ring import TruncRing, ring_matmul_batch

CD01_NAME = "Cramer-Damgard (CRYPTO'01)"
CKP_NAME = "Cramer-Kiltz-Padro / generalized Newton"


def _lift_matrix(abb, ring, A, C):
    n = A.shape[0]
    shares = np.zeros((abb.n, ring.D, n, n), dtype=abb.gf.dtype)
    shares[:, 1, :, :] = A.shares
    shares[0, 0, :, :] ^= C
    return Shared(abb, shares)


def _preconditioner(gf, n):
    q = gf.q
    if n - 1 > q - 1:
        raise ValueError("field too small for the diagonal preconditioner")
    alphas = np.array([gf.exp[i] for i in range(n - 1)], dtype=gf.dtype)
    C = np.zeros((n, n), dtype=gf.dtype)
    for i in range(1, n):
        C[i, i] = alphas[i - 1]
    return C


def _powers(abb, ring, B, K):
    powers = {1: B}
    cur = B
    for j in range(2, K + 1):
        cur = ring_matmul_batch(abb, ring, [(cur, B)])[0]
        powers[j] = cur
    return powers


def _powers_bsgs(abb, ring, B, K):
    mm = max(1, int(math.isqrt(K)))
    baby = {1: B}
    cur = B
    for j in range(2, mm + 1):
        cur = ring_matmul_batch(abb, ring, [(cur, B)])[0]
        baby[j] = cur
    giant = {}
    step = baby[mm]
    cur = step
    gg = (K + mm - 1) // mm
    for j in range(1, gg + 1):
        giant[j] = cur
        if j < gg:
            cur = ring_matmul_batch(abb, ring, [(cur, step)])[0]
    return baby, giant, mm, gg


def _traces_from_powers(abb, ring, baby, giant, mm, K, n):
    g = abb.gf
    masks = {}
    opens = []
    order = []
    for tag, d in (("b", baby), ("g", giant)):
        for j, X in d.items():
            t0 = time.perf_counter()
            M = ring.rand(abb.rng, (n, n))
            abb.prep += M.size
            sM = abb.share(M)
            abb.prep_time += time.perf_counter() - t0
            masks[(tag, j)] = (M, sM)
            opens.append(Shared(abb, X.shares ^ sM.shares))
            order.append((tag, j))
    vals = abb.open_many(opens)
    opened = {order[i]: vals[i] for i in range(len(order))}

    traces = {}
    for j, X in baby.items():
        if j <= K:
            traces[j] = _single_trace(abb, ring, opened[("b", j)], masks[("b", j)][1])
    for jg, G in giant.items():
        base_exp = jg * mm
        if base_exp <= K and base_exp not in traces:
            traces[base_exp] = _single_trace(abb, ring, opened[("g", jg)], masks[("g", jg)][1])
        for jb, Bp in baby.items():
            e = base_exp + jb
            if e > K or e in traces:
                continue
            traces[e] = _cross_trace(abb, ring, opened[("b", jb)], masks[("b", jb)],
                                     opened[("g", jg)], masks[("g", jg)], n)
    return traces


def _single_trace(abb, ring, dv, sM):
    out = np.zeros((abb.n, ring.D), dtype=abb.gf.dtype)
    for i in range(abb.n):
        out[i] = ring.trace(sM.shares[i])
    out[0] ^= ring.trace(dv)
    return Shared(abb, out)


def _cross_trace(abb, ring, dx, mx, dy, my, n):
    g = abb.gf
    Mx, sMx = mx
    My, sMy = my
    t0 = time.perf_counter()
    cross_val = ring.trace_of_product(Mx, My)
    abb.prep += ring.D
    scross = abb.share(cross_val)
    abb.prep_time += time.perf_counter() - t0
    base = ring.trace_of_product(dx, dy)
    out = np.zeros((abb.n, ring.D), dtype=g.dtype)
    for i in range(abb.n):
        out[i] = ring.trace_of_product(dx, sMy.shares[i]) ^ ring.trace_of_product(sMx.shares[i], dy)
    out ^= scross.shares
    out[0] ^= base
    return Shared(abb, out)


def newton_det(abb, A, n, mode="bsgs"):
    g = abb.gf
    D = n + 1
    ring = TruncRing(g, D)
    C = _preconditioner(g, n)
    B = _lift_matrix(abb, ring, A, C)
    K = 2 * n - 1
    if mode == "bsgs":
        baby, giant, mm, gg = _powers_bsgs(abb, ring, B, K)
    else:
        allp = _powers(abb, ring, B, K)
        baby, giant, mm, gg = allp, {}, K + 1, 0
    traces = _traces_from_powers(abb, ring, baby, giant, mm, K, n)

    rows = n - 1
    Pm = np.zeros((abb.n, D, rows, rows), dtype=g.dtype)
    Rm = np.zeros((abb.n, D, rows, 2), dtype=g.dtype)
    for a in range(rows):
        for c in range(rows):
            Pm[:, :, a, c] = traces[a + c + 1].shares
        Rm[:, :, a, 0] = traces[a + rows + 1].shares
        Rm[:, :, a, 1] = traces[a + rows + 2].shares
    Pms = Shared(abb, Pm)
    Rms = Shared(abb, Rm)

    e1 = traces[1]
    rhs_vec = np.zeros((abb.n, D, 2, 1), dtype=g.dtype)
    rhs_vec[:, :, 0, 0] = e1.shares
    rhs_vec[0, 0, 1, 0] ^= g.dtype(1)
    rhs = Shared(abb, rhs_vec)

    t0 = time.perf_counter()
    while True:
        S = ring.rand(abb.rng, (rows, rows))
        if g.inv_matrix(S[0]) is not None:
            break
    abb.prep += S.size
    sS = abb.share(S)
    abb.prep_time += time.perf_counter() - t0
    T = ring_matmul_batch(abb, ring, [(sS, Pms)])[0]
    Tv = abb.open(T)
    Tinv = ring.inv_matrix(Tv)
    Pinv = Shared(abb, np.stack([ring.matmul(Tinv, sS.shares[i]) for i in range(abb.n)]))
    w = ring_matmul_batch(abb, ring, [(Rms, rhs)])[0]
    u = ring_matmul_batch(abb, ring, [(Pinv, w)])[0]
    det_shares = u.shares[:, n, 0, 0]
    return Shared(abb, np.ascontiguousarray(det_shares))
