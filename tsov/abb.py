import time

import numpy as np

from .gf import GF


class Shared:
    __slots__ = ("shares", "abb", "shape", "_nb")

    def __init__(self, abb, shares):
        self.abb = abb
        self.shares = shares
        self.shape = shares.shape[1:]
        self._nb = shares.nbytes
        abb.live_bytes += self._nb
        if abb.live_bytes > abb.peak_bytes:
            abb.peak_bytes = abb.live_bytes

    def __del__(self):
        try:
            self.abb.live_bytes -= self._nb
        except Exception:
            pass

    @property
    def value(self):
        out = self.shares[0].copy()
        for i in range(1, self.shares.shape[0]):
            out ^= self.shares[i]
        return out

    def __add__(self, other):
        if isinstance(other, Shared):
            return Shared(self.abb, self.shares ^ other.shares)
        return self.abb.add_public(self, other)

    def __sub__(self, other):
        return self.__add__(other)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            k = (slice(None),) + key
        else:
            k = (slice(None), key)
        return Shared(self.abb, np.ascontiguousarray(self.shares[k]))

    def reshape(self, *shape):
        return Shared(self.abb, self.shares.reshape((self.shares.shape[0],) + tuple(shape)))

    def transpose(self):
        return Shared(self.abb, np.ascontiguousarray(np.swapaxes(self.shares, 1, 2)))


class _Pending:
    __slots__ = ("kind", "a", "b", "c", "x", "y", "dx", "dy", "out")

    def __init__(self, kind, a, b, c, x, y):
        self.kind = kind
        self.a = a
        self.b = b
        self.c = c
        self.x = x
        self.y = y
        self.dx = None
        self.dy = None
        self.out = None


class ABB:
    def __init__(self, gf, n_parties=3, seed=0):
        self.gf = gf if isinstance(gf, GF) else GF(gf)
        self.n = n_parties
        self.rng = np.random.default_rng(seed)
        self.reset_counters()
        self._pending = []

    def reset_counters(self):
        self.rounds = 0
        self.comm = 0
        self.prep = 0
        self.mults = 0
        self.alloc = 0
        self.prep_time = 0.0
        self.live_bytes = 0
        self.peak_bytes = 0

    def _account(self, nelem):
        self.alloc += nelem

    def share(self, value):
        value = np.asarray(value, dtype=self.gf.dtype)
        shares = self.gf.rand(self.rng, (self.n,) + value.shape)
        acc = shares[0].copy()
        for i in range(1, self.n):
            acc ^= shares[i]
        shares[0] ^= acc ^ value
        self._account(value.size)
        return Shared(self, shares)

    def zero(self, shape):
        shares = np.zeros((self.n,) + tuple(shape), dtype=self.gf.dtype)
        self._account(int(np.prod(shape)) if shape else 1)
        return Shared(self, shares)

    def rand(self, shape):
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        shares = self.gf.rand(self.rng, (self.n,) + shape)
        self.prep += int(np.prod(shape))
        self._account(int(np.prod(shape)))
        return Shared(self, shares)

    def rand_nonzero(self, shape):
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        v = self.gf.rand_nonzero(self.rng, shape)
        self.prep += int(np.prod(shape))
        return self.share(v)

    def add_public(self, x, c):
        shares = x.shares.copy()
        shares[0] = shares[0] ^ np.asarray(c, dtype=self.gf.dtype)
        return Shared(self, shares)

    def mul_public(self, x, c):
        c = np.asarray(c, dtype=self.gf.dtype)
        out = np.stack([self.gf.mul(x.shares[i], c) for i in range(self.n)])
        return Shared(self, out)

    def matmul_public_left(self, C, x):
        C = np.asarray(C, dtype=self.gf.dtype)
        out = np.stack([self.gf.matmul(C, x.shares[i]) for i in range(self.n)])
        return Shared(self, out)

    def matmul_public_right(self, x, C):
        C = np.asarray(C, dtype=self.gf.dtype)
        out = np.stack([self.gf.matmul(x.shares[i], C) for i in range(self.n)])
        return Shared(self, out)

    def open(self, x):
        return self.open_many([x])[0]

    def open_many(self, xs):
        self.rounds += 1
        vals = []
        for x in xs:
            self.comm += x.shares[0].size
            vals.append(x.value)
        return vals

    def _triple(self, kind, sx, sy):
        t0 = time.perf_counter()
        a = self.gf.rand(self.rng, sx)
        b = self.gf.rand(self.rng, sy)
        if kind == "mm":
            c = self.gf.matmul(a, b)
        elif kind == "sm":
            c = self.gf.mul(a, b)
        elif kind == "outer":
            c = self.gf.matmul(a.reshape(-1, 1), b.reshape(1, -1))
        elif kind == "mv":
            c = self.gf.matmul(a, b.reshape(-1, 1)).reshape(-1)
        elif kind == "el":
            c = self.gf.mul(a, b)
        else:
            raise ValueError(kind)
        self.prep += a.size + b.size + c.size
        res = (self.share(a), self.share(b), self.share(c))
        self.prep_time += time.perf_counter() - t0
        return res

    def mul_begin(self, x, y, kind):
        a, b, c = self._triple(kind, x.shape, y.shape)
        p = _Pending(kind, a, b, c, x, y)
        p.dx = Shared(self, x.shares ^ a.shares)
        p.dy = Shared(self, y.shares ^ b.shares)
        self._pending.append(p)
        return p

    def flush(self):
        if not self._pending:
            return
        items = []
        for p in self._pending:
            items.append(p.dx)
            items.append(p.dy)
        vals = self.open_many(items)
        for k, p in enumerate(self._pending):
            dx = vals[2 * k]
            dy = vals[2 * k + 1]
            p.out = self._combine(p, dx, dy)
        self._pending = []

    def _combine(self, p, dx, dy):
        g = self.gf
        kind = p.kind
        if kind == "mm":
            self.mults += dx.shape[0] * dx.shape[1] * dy.shape[1]
            base = g.matmul(dx, dy)
            t1 = self.matmul_public_left(dx, p.b)
            t2 = self.matmul_public_right(p.a, dy)
            out = t1.shares ^ t2.shares ^ p.c.shares
            out[0] ^= base
            return Shared(self, out)
        if kind == "mv":
            self.mults += dx.shape[0] * dx.shape[1]
            base = g.matmul(dx, dy.reshape(-1, 1)).reshape(-1)
            t1 = Shared(self, np.stack([g.matmul(dx, p.b.shares[i].reshape(-1, 1)).reshape(-1)
                                        for i in range(self.n)]))
            t2 = Shared(self, np.stack([g.matmul(p.a.shares[i], dy.reshape(-1, 1)).reshape(-1)
                                        for i in range(self.n)]))
            out = t1.shares ^ t2.shares ^ p.c.shares
            out[0] ^= base
            return Shared(self, out)
        if kind == "sm":
            self.mults += int(np.prod(p.y.shape))
            base = g.mul(dx, dy)
            t1 = self.mul_public(p.b, dx)
            t2 = self.mul_public(p.a, dy)
            out = t1.shares ^ t2.shares ^ p.c.shares
            out[0] ^= base
            return Shared(self, out)
        if kind == "el":
            self.mults += int(np.prod(p.x.shape))
            base = g.mul(dx, dy)
            t1 = self.mul_public(p.b, dx)
            t2 = self.mul_public(p.a, dy)
            out = t1.shares ^ t2.shares ^ p.c.shares
            out[0] ^= base
            return Shared(self, out)
        if kind == "outer":
            self.mults += dx.size * dy.size
            base = g.matmul(dx.reshape(-1, 1), dy.reshape(1, -1))
            t1 = Shared(self, np.stack([g.matmul(dx.reshape(-1, 1), p.b.shares[i].reshape(1, -1))
                                        for i in range(self.n)]))
            t2 = Shared(self, np.stack([g.matmul(p.a.shares[i].reshape(-1, 1), dy.reshape(1, -1))
                                        for i in range(self.n)]))
            out = t1.shares ^ t2.shares ^ p.c.shares
            out[0] ^= base
            return Shared(self, out)
        raise ValueError(kind)

    def matmul(self, x, y):
        p = self.mul_begin(x, y, "mm")
        self.flush()
        return p.out

    def matvec(self, x, y):
        p = self.mul_begin(x, y, "mv")
        self.flush()
        return p.out

    def scalmul(self, s, m):
        p = self.mul_begin(s, m, "sm")
        self.flush()
        return p.out

    def elmul(self, x, y):
        p = self.mul_begin(x, y, "el")
        self.flush()
        return p.out

    def outer(self, u, v):
        p = self.mul_begin(u, v, "outer")
        self.flush()
        return p.out

    def stats(self):
        return {
            "rounds": self.rounds,
            "comm": self.comm,
            "prep": self.prep,
            "mults": self.mults,
            "alloc": self.alloc,
            "peak_shared": self.peak_bytes,
            "prep_time": self.prep_time,
        }
