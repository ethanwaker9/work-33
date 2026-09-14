import time

import numpy as np

from .abb import Shared


class TruncRing:
    def __init__(self, gf, D):
        self.gf = gf
        self.D = D

    def zeros(self, shape):
        return np.zeros((self.D,) + tuple(shape), dtype=self.gf.dtype)

    def rand(self, rng, shape):
        return self.gf.rand(rng, (self.D,) + tuple(shape))

    def embed(self, M, deg=0):
        out = self.zeros(M.shape)
        out[deg] = M
        return out

    def matmul(self, A, B):
        g = self.gf
        D = self.D
        n = A.shape[1]
        p = B.shape[2]
        C = np.zeros((D, n, p), dtype=g.dtype)
        for i in range(D):
            Ai = A[i]
            if not Ai.any():
                continue
            for j in range(D - i):
                Bj = B[j]
                if not Bj.any():
                    continue
                C[i + j] ^= g.matmul(Ai, Bj)
        return C

    def mul(self, a, b):
        g = self.gf
        D = self.D
        out = np.zeros((D,) + a.shape[1:], dtype=g.dtype)
        for i in range(D):
            if not np.any(a[i]):
                continue
            for j in range(D - i):
                out[i + j] ^= g.mul(a[i], b[j])
        return out

    def trace_of_product(self, X, Y):
        g = self.gf
        D = self.D
        out = np.zeros(D, dtype=g.dtype)
        YT = np.swapaxes(Y, 1, 2)
        for i in range(D):
            Xi = X[i]
            if not Xi.any():
                continue
            for j in range(D - i):
                Yj = YT[j]
                if not Yj.any():
                    continue
                out[i + j] ^= np.bitwise_xor.reduce(g.mul(Xi, Yj).reshape(-1))
        return out

    def trace(self, A):
        return np.array([np.bitwise_xor.reduce(np.diagonal(A[d])) for d in range(self.D)],
                        dtype=self.gf.dtype)

    def inv_matrix(self, M):
        g = self.gf
        D = self.D
        n = M.shape[1]
        M0inv = g.inv_matrix(M[0])
        if M0inv is None:
            return None
        X = np.zeros((D, n, n), dtype=g.dtype)
        X[0] = M0inv
        for d in range(1, D):
            acc = np.zeros((n, n), dtype=g.dtype)
            for i in range(1, d + 1):
                if M[i].any() and X[d - i].any():
                    acc ^= g.matmul(M[i], X[d - i])
            X[d] = g.matmul(M0inv, acc)
        return X

    def rand_invertible(self, rng, n):
        g = self.gf
        while True:
            M = self.rand(rng, (n, n))
            if g.inv_matrix(M[0]) is not None:
                return M, self.inv_matrix(M)


def ring_share(abb, ring, M):
    return abb.share(M)


def ring_rand(abb, ring, shape):
    return abb.rand((ring.D,) + tuple(shape))


def _pub_ring_matmul_shared(abb, ring, P, X, left=True):
    out = []
    for i in range(abb.n):
        out.append(ring.matmul(P, X.shares[i]) if left else ring.matmul(X.shares[i], P))
    return Shared(abb, np.stack(out))


def ring_matmul_begin(abb, ring, X, Y):
    t0 = time.perf_counter()
    A = ring.rand(abb.rng, X.shape[1:])
    B = ring.rand(abb.rng, Y.shape[1:])
    C = ring.matmul(A, B)
    abb.prep += A.size + B.size + C.size
    sA, sB, sC = abb.share(A), abb.share(B), abb.share(C)
    abb.prep_time += time.perf_counter() - t0
    dX = Shared(abb, X.shares ^ sA.shares)
    dY = Shared(abb, Y.shares ^ sB.shares)
    return {"dX": dX, "dY": dY, "A": sA, "B": sB, "C": sC, "ring": ring}


def ring_matmul_finish(abb, pend, dx, dy):
    ring = pend["ring"]
    base = ring.matmul(dx, dy)
    t1 = _pub_ring_matmul_shared(abb, ring, dx, pend["B"], left=True)
    t2 = _pub_ring_matmul_shared(abb, ring, dy, pend["A"], left=False)
    out = t1.shares ^ t2.shares ^ pend["C"].shares
    out[0] ^= base
    return Shared(abb, out)


def ring_matmul_batch(abb, ring, pairs):
    pends = [ring_matmul_begin(abb, ring, X, Y) for X, Y in pairs]
    items = []
    for p in pends:
        items.append(p["dX"])
        items.append(p["dY"])
    vals = abb.open_many(items)
    res = []
    for k, p in enumerate(pends):
        res.append(ring_matmul_finish(abb, p, vals[2 * k], vals[2 * k + 1]))
    return res
