import time

import numpy as np

from .abb import Shared


class MaskedMatrix:
    def __init__(self, abb, X, batched=None):
        self.abb = abb
        g = abb.gf
        t0 = time.perf_counter()
        A = g.rand(abb.rng, X.shape)
        abb.prep += A.size
        self.sA = abb.share(A)
        abb.prep_time += time.perf_counter() - t0
        self.shape = X.shape
        self._masked = Shared(abb, X.shares ^ self.sA.shares)
        if batched is None:
            self.d = abb.open(self._masked)
        else:
            batched.append(self)

    @staticmethod
    def batch(abb, mats):
        pend = []
        out = [MaskedMatrix(abb, X, batched=pend) for X in mats]
        if pend:
            vals = abb.open_many([m._masked for m in pend])
            for m, v in zip(pend, vals):
                m.d = v
        return out


class _Job:
    __slots__ = ("mm", "y", "dy", "sAC")


def masked_mul_begin(abb, mm, Y):
    g = abb.gf
    two = len(Y.shape) == 2
    t0 = time.perf_counter()
    C = g.rand(abb.rng, Y.shape)
    Cm = C if two else C.reshape(-1, 1)
    AC = np.stack([g.matmul(mm.sA.shares[i], Cm) for i in range(abb.n)])
    if not two:
        AC = AC.reshape(abb.n, -1)
    abb.prep += C.size + AC[0].size
    sC = abb.share(C)
    abb.prep_time += time.perf_counter() - t0
    j = _Job()
    j.mm = mm
    j.y = Y
    j.sAC = Shared(abb, AC)
    j.dy = Shared(abb, Y.shares ^ sC.shares)
    return j


def masked_mul_finish(abb, j, dyv):
    g = abb.gf
    mm = j.mm
    two = len(j.y.shape) == 2
    cols = j.y.shape[1] if two else 1
    abb.mults += mm.shape[0] * mm.shape[1] * cols
    dm = dyv if two else dyv.reshape(-1, 1)
    out = []
    for i in range(abb.n):
        ys = j.y.shares[i] if two else j.y.shares[i].reshape(-1, 1)
        v = g.matmul(mm.d, ys) ^ g.matmul(mm.sA.shares[i], dm)
        out.append(v if two else v.reshape(-1))
    res = np.stack(out) ^ j.sAC.shares
    return Shared(abb, res)


def run_masked(abb, jobs):
    pends = [masked_mul_begin(abb, mm, Y) for mm, Y in jobs]
    if not pends:
        return []
    vals = abb.open_many([p.dy for p in pends])
    return [masked_mul_finish(abb, pends[i], vals[i]) for i in range(len(pends))]
