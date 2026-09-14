import numpy as np

from .common import berkowitz_det, zero_test
from .masking import _kernel_basis, _right_inverse
from .newton import newton_det

BERK_NAME = "Samuelson-Berkowitz (secure det)"
NEWTON_NAME = "Generalized Newton (secure det)"


def _solve_with_det(abb, A, y, s, t, det_fn, max_trials=3):
    g = abb.gf
    trials = 0
    for _ in range(max_trials):
        trials += 1
        R = abb.rand((s, s))
        S = abb.rand((t, t))
        RA = abb.matmul(R, A)
        T = abb.matmul(RA, S)
        cols = list(abb.rng.permutation(t)[:s])
        sub = T[:, cols]
        delta = det_fn(abb, sub, s)
        if not zero_test(abb, delta):
            continue
        Tv = abb.open(T)
        Tri = _right_inverse(g, Tv, s, t)
        if Tri is None:
            continue
        ST = abb.matmul_public_right(S, Tri)
        Ainv = abb.matmul(ST, R)
        x = abb.matvec(Ainv, y)
        if t > s:
            K = _kernel_basis(g, Tv, t)
            if K.shape[1] > 0:
                zz = abb.rand((K.shape[1],))
                z = abb.matmul_public_left(K, zz.reshape(K.shape[1], 1)).reshape(t)
                x = x + abb.matvec(S, z)
        return {"ok": True, "x": x, "trials": trials, "leak": "none"}
    return {"ok": False, "x": None, "trials": trials, "leak": "none"}


def solve_berkowitz(abb, A, y, s, t, max_trials=3):
    return _solve_with_det(abb, A, y, s, t,
                           lambda ab, M, n: berkowitz_det(ab, M, n), max_trials)


def solve_newton(abb, A, y, s, t, max_trials=3):
    return _solve_with_det(abb, A, y, s, t,
                           lambda ab, M, n: newton_det(ab, M, n, "bsgs"), max_trials)


def solve_cd01(abb, A, y, s, t, max_trials=3):
    return _solve_with_det(abb, A, y, s, t,
                           lambda ab, M, n: newton_det(ab, M, n, "all"), max_trials)
