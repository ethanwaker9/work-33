import time

import numpy as np

from ..abb import Shared
from ..masked import MaskedMatrix


def zero_test(abb, delta):
    rho = abb.rand_nonzero(())
    prod = abb.elmul(delta, rho)
    return int(abb.open(prod)) != 0


class _SharedInputJob:
    __slots__ = ("ops", "v", "dv", "prods")


def apply_ops_begin(abb, ops, v):
    g = abb.gf
    t0 = time.perf_counter()
    c = g.rand(abb.rng, v.shape)
    cm = c.reshape(-1, 1)
    prods = []
    for op in ops:
        AC = np.stack([g.matmul(op.sA.shares[i], cm).reshape(-1) for i in range(abb.n)])
        abb.prep += AC[0].size
        prods.append(Shared(abb, AC))
    abb.prep += c.size
    sc = abb.share(c)
    abb.prep_time += time.perf_counter() - t0
    j = _SharedInputJob()
    j.ops = ops
    j.v = v
    j.prods = prods
    j.dv = Shared(abb, v.shares ^ sc.shares)
    return j


def apply_ops_finish(abb, j, dv):
    g = abb.gf
    dm = dv.reshape(-1, 1)
    out = []
    for k, op in enumerate(j.ops):
        abb.mults += op.shape[0] * op.shape[1]
        res = []
        for i in range(abb.n):
            res.append((g.matmul(op.d, j.v.shares[i].reshape(-1, 1))
                        ^ g.matmul(op.sA.shares[i], dm)).reshape(-1))
        out.append(Shared(abb, np.stack(res) ^ j.prods[k].shares))
    return out


def run_ops(abb, jobs):
    pends = [apply_ops_begin(abb, ops, v) for ops, v in jobs]
    if not pends:
        return []
    vals = abb.open_many([p.dv for p in pends])
    return [apply_ops_finish(abb, pends[i], vals[i]) for i in range(len(pends))]


def _toeplitz_shares(abb, col_shares, rows, cols):
    out = np.zeros((abb.n, rows, cols), dtype=abb.gf.dtype)
    L = col_shares.shape[-1]
    for j in range(cols):
        length = min(rows - j, L)
        if length > 0:
            out[:, j:j + length, j] = col_shares[:, :length]
    return out


def _toeplitz_public(gf, col, rows, cols):
    out = np.zeros((rows, cols), dtype=gf.dtype)
    L = col.shape[0]
    for j in range(cols):
        length = min(rows - j, L)
        if length > 0:
            out[j:j + length, j] = col[:length]
    return out


def berkowitz_charpoly(abb, A, n):
    g = abb.gf
    if n == 1:
        col = np.zeros((abb.n, 2), dtype=g.dtype)
        col[0, 0] = 1
        col[:, 1] = A[0, 0].shares
        return Shared(abb, col)

    levels = []
    for i in range(n):
        Ai = A[i:, i:] if i > 0 else A
        a = Ai[0, 0]
        if n - i == 1:
            levels.append({"a": a, "len": 2, "vals": [], "M": None})
            continue
        levels.append({"a": a, "len": n - i + 1, "vals": [],
                       "M": Ai[1:, 1:], "R": Ai[0, 1:], "S": Ai[1:, 0]})

    ops = {}
    state = {}
    tomask = []
    keys = []
    for idx, lv in enumerate(levels):
        if lv["M"] is None:
            continue
        tomask.append(lv["M"])
        tomask.append(lv["R"].reshape(1, -1))
        keys.append(idx)
        state[idx] = lv["S"]
    masked = MaskedMatrix.batch(abb, tomask)
    for k, idx in enumerate(keys):
        ops[idx] = (masked[2 * k], masked[2 * k + 1])

    nsteps = max((lv["len"] - 2 for lv in levels), default=0)
    for step in range(nsteps):
        jobs = []
        active = []
        for idx, lv in enumerate(levels):
            if lv["M"] is None or step > lv["len"] - 3:
                continue
            mm, mr = ops[idx]
            need_next = step < lv["len"] - 3
            jobs.append(((mm, mr) if need_next else (mr,), state[idx]))
            active.append((idx, need_next))
        if not jobs:
            break
        res = run_ops(abb, jobs)
        for k, (idx, need_next) in enumerate(active):
            if need_next:
                state[idx] = res[k][0]
                levels[idx]["vals"].append(res[k][1].reshape())
            else:
                levels[idx]["vals"].append(res[k][0].reshape())

    cols = []
    for lv in levels:
        L = lv["len"]
        col = np.zeros((abb.n, L), dtype=g.dtype)
        col[0, 0] = 1
        col[:, 1] = lv["a"].shares
        for j, e in enumerate(lv["vals"][:L - 2]):
            col[:, j + 2] = e.shares
        cols.append(Shared(abb, col))

    t0 = time.perf_counter()
    rmask = {}
    for i in range(n - 1):
        L = levels[i]["len"]
        rmask[i] = g.rand(abb.rng, (L,))
        abb.prep += L
    srs = {i: abb.share(rmask[i]) for i in rmask}
    abb.prep_time += time.perf_counter() - t0
    dcols = {}
    if rmask:
        vals = abb.open_many([Shared(abb, cols[i].shares ^ srs[i].shares) for i in range(n - 1)])
        for i in range(n - 1):
            dcols[i] = vals[i]

    p = None
    for i in range(n - 1, -1, -1):
        L = levels[i]["len"]
        if p is None:
            p = Shared(abb, np.ascontiguousarray(cols[i].shares[:, :L]))
            continue
        t0 = time.perf_counter()
        cvec = g.rand(abb.rng, (L - 1,))
        Ar = _toeplitz_public(g, rmask[i], L, L - 1)
        Arc = g.matmul(Ar, cvec.reshape(-1, 1)).reshape(-1)
        abb.prep += (L - 1) + L
        sr = srs[i]
        sc = abb.share(cvec)
        sArc = abb.share(Arc)
        abb.prep_time += time.perf_counter() - t0
        dcol = dcols[i]
        dp = abb.open(Shared(abb, p.shares ^ sc.shares))
        Dpub = _toeplitz_public(g, dcol, L, L - 1)
        Ash = _toeplitz_shares(abb, sr.shares, L, L - 1)
        acc = []
        for k in range(abb.n):
            acc.append((g.matmul(Dpub, p.shares[k].reshape(-1, 1))
                        ^ g.matmul(Ash[k], dp.reshape(-1, 1))).reshape(-1))
        abb.mults += L * (L - 1)
        p = Shared(abb, np.stack(acc) ^ sArc.shares)
    return p


def berkowitz_det(abb, A, n):
    return berkowitz_charpoly(abb, A, n)[n]


def small_det_adj(abb, M, b):
    g = abb.gf
    if b == 1:
        return M[0, 0], abb.share(np.ones((1, 1), dtype=g.dtype))
    idx = [(i, j) for i in range(b) for j in range(i + 1, b)]
    left = []
    right = []
    for (i, j) in idx:
        for (k, l) in idx:
            left.append(M[i, k])
            right.append(M[j, l])
            left.append(M[i, l])
            right.append(M[j, k])
    lv = _stack(abb, left)
    rv = _stack(abb, right)
    prod = abb.elmul(lv, rv)
    minors = {}
    pos = 0
    for (i, j) in idx:
        for (k, l) in idx:
            minors[(i, j, k, l)] = Shared(abb, np.ascontiguousarray(prod.shares[:, pos])) + \
                Shared(abb, np.ascontiguousarray(prod.shares[:, pos + 1]))
            pos += 2
    if b == 2:
        det = minors[(0, 1, 0, 1)]
        adj = np.zeros((abb.n, 2, 2), dtype=g.dtype)
        adj[:, 0, 0] = M[1, 1].shares
        adj[:, 0, 1] = M[0, 1].shares
        adj[:, 1, 0] = M[1, 0].shares
        adj[:, 1, 1] = M[0, 0].shares
        return det, Shared(abb, adj)
    if b == 3:
        left = []
        right = []
        for j in range(3):
            rest = [c for c in range(3) if c != j]
            left.append(M[0, j])
            right.append(minors[(1, 2, rest[0], rest[1])])
        det = _sum_products(abb, left, right)
        adj = np.zeros((abb.n, 3, 3), dtype=g.dtype)
        for i in range(3):
            for j in range(3):
                ri = [r for r in range(3) if r != i]
                cj = [c for c in range(3) if c != j]
                adj[:, j, i] = minors[(ri[0], ri[1], cj[0], cj[1])].shares
        return det, Shared(abb, adj)
    if b == 4:
        left = []
        right = []
        for (k, l) in idx:
            comp = [c for c in range(4) if c not in (k, l)]
            left.append(minors[(0, 1, k, l)])
            right.append(minors[(2, 3, comp[0], comp[1])])
        keys = []
        for i in range(4):
            for j in range(4):
                ri = [r for r in range(4) if r != i]
                cj = [c for c in range(4) if c != j]
                for t in range(3):
                    cc = [c for c in cj if c != cj[t]]
                    left.append(M[ri[0], cj[t]])
                    right.append(minors[(ri[1], ri[2], cc[0], cc[1])])
                keys.append((i, j))
        lv2 = _stack(abb, left)
        rv2 = _stack(abb, right)
        pr = abb.elmul(lv2, rv2)
        acc = np.zeros((abb.n,), dtype=g.dtype)
        for u in range(6):
            acc ^= pr.shares[:, u]
        det = Shared(abb, acc)
        adj = np.zeros((abb.n, 4, 4), dtype=g.dtype)
        for t, (i, j) in enumerate(keys):
            a2 = np.zeros((abb.n,), dtype=g.dtype)
            for u in range(3):
                a2 ^= pr.shares[:, 6 + 3 * t + u]
            adj[:, j, i] = a2
        return det, Shared(abb, adj)
    raise ValueError("small_det_adj supports b <= 4")


def _stack(abb, items):
    return Shared(abb, np.stack([it.shares for it in items], axis=1))


def _sum_products(abb, left, right):
    lv = _stack(abb, left)
    rv = _stack(abb, right)
    pr = abb.elmul(lv, rv)
    acc = np.zeros((abb.n,), dtype=abb.gf.dtype)
    for u in range(len(left)):
        acc ^= pr.shares[:, u]
    return Shared(abb, acc)


def det_adj(abb, M, b):
    if b <= 4:
        return small_det_adj(abb, M, b)
    p = berkowitz_charpoly(abb, M, b)
    det = p[b]
    powers = [None, M]
    cur = M
    for k in range(2, b):
        cur = abb.matmul(cur, M)
        powers.append(cur)
    acc = powers[b - 1]
    for k in range(1, b - 1):
        acc = acc + abb.scalmul(p[k], powers[b - 1 - k])
    acc = acc + abb.scalmul(p[b - 1], abb.share(np.eye(b, dtype=abb.gf.dtype)))
    return det, acc
