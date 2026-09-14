import hashlib

import numpy as np

from .gf import GF

PARAMS = {
    "UOV-Ip": {"n": 112, "m": 44, "o": 44, "deg": 8},
    "UOV-III": {"n": 184, "m": 72, "o": 72, "deg": 8},
    "UOV-V": {"n": 244, "m": 96, "o": 96, "deg": 8},
    "UOV-Is": {"n": 160, "m": 64, "o": 64, "deg": 4},
    "TEST": {"n": 28, "m": 11, "o": 11, "deg": 8},
    "MAYO-1": {"n": 66, "m": 64, "o": 8, "k": 9, "deg": 4},
    "MAYO-2": {"n": 78, "m": 64, "o": 18, "k": 4, "deg": 4},
    "MAYO-3": {"n": 99, "m": 96, "o": 10, "k": 11, "deg": 4},
    "MAYO-5": {"n": 133, "m": 128, "o": 12, "k": 12, "deg": 4},
}


def upper(gf, M):
    out = np.zeros_like(M)
    n = M.shape[0]
    iu = np.triu_indices(n, 1)
    out[np.diag_indices(n)] = M[np.diag_indices(n)]
    out[iu] = M[iu] ^ M.T[iu]
    return out


class UOV:
    def __init__(self, name, seed=0):
        p = PARAMS[name]
        self.name = name
        self.n = p["n"]
        self.m = p["m"]
        self.o = p["o"]
        self.k = p.get("k", 1)
        self.gf = GF(p["deg"])
        self.v = self.n - self.o
        self.rng = np.random.default_rng(seed)

    def public_seed_material(self, rng=None):
        g = self.gf
        rng = rng or self.rng
        P1 = np.stack([np.triu(g.rand(rng, (self.v, self.v))) for _ in range(self.m)])
        P2 = np.stack([g.rand(rng, (self.v, self.o)) for _ in range(self.m)])
        return P1, P2

    def keygen(self, rng=None):
        g = self.gf
        rng = rng or self.rng
        P1, P2 = self.public_seed_material(rng)
        O = g.rand(rng, (self.v, self.o))
        P3 = self.derive_p3(P1, P2, O)
        return (P1, P2, P3), O

    def derive_p3(self, P1, P2, O):
        g = self.gf
        out = []
        for i in range(self.m):
            M = g.matmul(g.matmul(O.T, P1[i]), O) ^ g.matmul(O.T, P2[i])
            out.append(upper(g, M))
        return np.stack(out)

    def hash_target(self, msg, salt):
        g = self.gf
        h = hashlib.shake_256(msg + salt).digest(self.m * 2)
        arr = np.frombuffer(h, dtype=np.uint8)[: self.m].astype(np.uint16)
        return (arr % np.uint16(g.q)).astype(g.dtype)

    def build_system(self, P1, P2, O, v):
        g = self.gf
        rows = []
        consts = []
        for i in range(self.m):
            S = g.matmul(P1[i] ^ P1[i].T, O) ^ P2[i]
            rows.append(g.matmul(v.reshape(1, -1), S).reshape(-1))
            consts.append(int(g.matmul(v.reshape(1, -1), g.matmul(P1[i], v.reshape(-1, 1)))[0, 0]))
        A = np.stack(rows)
        w = np.array(consts, dtype=g.dtype)
        return A, w

    def sign(self, pk, O, msg, rng=None):
        g = self.gf
        rng = rng or self.rng
        P1, P2, P3 = pk
        for _ in range(64):
            salt = bytes(rng.integers(0, 256, size=24, dtype=np.uint64).astype(np.uint8))
            t = self.hash_target(msg, salt)
            vv = g.rand(rng, (self.v,))
            A, w = self.build_system(P1, P2, O, vv)
            y = t ^ w
            x = g.solve(A, y)
            if x is None:
                continue
            s = np.concatenate([vv ^ g.matvec(O, x), x])
            return s, salt
        raise RuntimeError("signing failed")

    def verify(self, pk, msg, sig):
        g = self.gf
        P1, P2, P3 = pk
        s, salt = sig
        t = self.hash_target(msg, salt)
        s1 = s[: self.v].reshape(-1, 1)
        s2 = s[self.v:].reshape(-1, 1)
        out = []
        for i in range(self.m):
            val = g.matmul(s1.T, g.matmul(P1[i], s1)) ^ g.matmul(s1.T, g.matmul(P2[i], s2)) \
                ^ g.matmul(s2.T, g.matmul(P3[i], s2))
            out.append(int(val[0, 0]))
        return np.array_equal(np.array(out, dtype=g.dtype), t)
