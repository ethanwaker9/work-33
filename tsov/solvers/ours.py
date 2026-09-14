import numpy as np

from .common import berkowitz_det, det_adj, zero_test

NAME = "BDFC (this work)"


def default_block(s):
    return min(4, max(1, s - 1)) if s > 1 else 1


def block_condense_det(abb, D, s, b):
    while s > b:
        bb = min(b, s - 1)
        B11 = D[:bb, :bb]
        B12 = D[:bb, bb:]
        B21 = D[bb:, :bb]
        B22 = D[bb:, bb:]
        d, adj = det_adj(abb, B11, bb)
        pU = abb.mul_begin(adj, B12, "mm")
        pW = abb.mul_begin(d, B22, "sm")
        abb.flush()
        V = abb.matmul(B21, pU.out)
        D = pW.out + V
        s -= bb
    if s <= 4:
        d, _ = det_adj(abb, D, s)
        return d
    return berkowitz_det(abb, D, s)


def solve(abb, A, y, s, t, block=None, max_trials=3):
    g = abb.gf
    b = default_block(s) if block is None else block
    trials = 0
    for _ in range(max_trials):
        trials += 1
        S = abb.rand((t, s))
        B = abb.matmul(A, S)
        delta = block_condense_det(abb, B, s, b)
        if not zero_test(abb, delta):
            continue
        Bv = abb.open(B)
        Binv = g.inv_matrix(Bv)
        z = abb.matmul_public_left(Binv, y.reshape(s, 1)).reshape(s)
        x = abb.matvec(S, z)
        if t > s:
            w = abb.rand((t,))
            Aw = abb.matvec(A, w)
            u = abb.matmul_public_left(Binv, Aw.reshape(s, 1)).reshape(s)
            x = x + w + abb.matvec(S, u)
        return {"ok": True, "x": x, "trials": trials, "block": b, "leak": "full-rank bit"}
    return {"ok": False, "x": None, "trials": trials, "block": b, "leak": "full-rank bit"}


def offline_online_split(abb, A, s, t, block=None, max_trials=3):
    g = abb.gf
    b = default_block(s) if block is None else block
    for _ in range(max_trials):
        S = abb.rand((t, s))
        B = abb.matmul(A, S)
        delta = block_condense_det(abb, B, s, b)
        if not zero_test(abb, delta):
            continue
        Bv = abb.open(B)
        Binv = g.inv_matrix(Bv)
        Gamma = abb.matmul_public_right(S, Binv)
        c = abb.zero((t,))
        if t > s:
            w = abb.rand((t,))
            Aw = abb.matvec(A, w)
            u = abb.matmul_public_left(Binv, Aw.reshape(s, 1)).reshape(s)
            c = w + abb.matvec(S, u)
        return {"ok": True, "Gamma": Gamma, "c": c, "block": b}
    return {"ok": False}


def _sub_rand(abb, shape, up, rng):
    idx = rng.integers(0, up.shape[0], size=shape, dtype=np.uint64)
    return abb.share(up[idx])


def solve_subfield(abb, A, y, s, t, up, block=None, max_trials=3, solve_trials=8):
    g = abb.gf
    b = default_block(s) if block is None else block
    trials = 0
    for _ in range(max_trials):
        trials += 1
        Sp = abb.rand((t, s))
        Bp = abb.matmul(A, Sp)
        delta = block_condense_det(abb, Bp, s, b)
        if not zero_test(abb, delta):
            continue
        for _ in range(solve_trials):
            S0 = _sub_rand(abb, (t, s), up, abb.rng)
            B = abb.matmul(A, S0)
            Bv = abb.open(B)
            Binv = g.inv_matrix(Bv)
            if Binv is None:
                continue
            z = abb.matmul_public_left(Binv, y.reshape(s, 1)).reshape(s)
            x = abb.matvec(S0, z)
            if t > s:
                w = _sub_rand(abb, (t,), up, abb.rng)
                Aw = abb.matvec(A, w)
                u = abb.matmul_public_left(Binv, Aw.reshape(s, 1)).reshape(s)
                x = x + w + abb.matvec(S0, u)
            return {"ok": True, "x": x, "trials": trials, "block": b,
                    "leak": "full-rank bit"}
        break
    return {"ok": False, "x": None, "trials": trials, "block": b,
            "leak": "full-rank bit"}
