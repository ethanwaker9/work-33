import time

import numpy as np

from .abb import Shared
from .masked import MaskedMatrix, run_masked
from .solvers import ours
from .solvers.common import run_ops
from .uov import upper


def quadratic_forms(abb, sv, P1):
    g = abb.gf
    m = len(P1)
    v = sv.shape[0]
    t0 = time.perf_counter()
    c = g.rand(abb.rng, (v,))
    qs = np.zeros((abb.n, m), dtype=g.dtype)
    sc = abb.share(c)
    for i in range(m):
        val = g.matmul(c.reshape(1, -1), g.matmul(P1[i], c.reshape(-1, 1)))[0, 0]
        qs[0, i] = val
    abb.prep += v + m
    sq = abb.share(qs[0])
    abb.prep_time += time.perf_counter() - t0
    d = abb.open(Shared(abb, sv.shares ^ sc.shares))
    out = np.zeros((abb.n, m), dtype=g.dtype)
    for i in range(m):
        Pd = g.matmul(P1[i], d.reshape(-1, 1)).reshape(-1)
        dP = g.matmul(d.reshape(1, -1), P1[i]).reshape(-1)
        for k in range(abb.n):
            out[k, i] = np.bitwise_xor.reduce(g.mul(sc.shares[k], Pd)) ^ \
                np.bitwise_xor.reduce(g.mul(dP, sc.shares[k]))
        out[0, i] ^= int(g.matmul(d.reshape(1, -1), Pd.reshape(-1, 1))[0, 0])
    abb.mults += m * v
    return Shared(abb, out ^ sq.shares)


class ThresholdUOV:
    def __init__(self, abb, uov):
        self.abb = abb
        self.uov = uov
        self.gf = abb.gf

    def dkg(self, P1, P2):
        abb = self.abb
        u = self.uov
        g = self.gf
        sO = abb.rand((u.v, u.o))
        mOT = MaskedMatrix(abb, sO.transpose())
        jobs = []
        for i in range(u.m):
            Z = abb.matmul_public_left(P1[i], sO)
            jobs.append((mOT, Z))
        prods = run_masked(abb, jobs)
        p3s = []
        for i in range(u.m):
            base = abb.matmul_public_right(sO.transpose(), P2[i])
            p3s.append(prods[i] + base)
        vals = abb.open_many(p3s)
        P3 = np.stack([upper(g, v) for v in vals])
        return sO, P3

    def verify_key(self, P1, P2, P3, sO, reps=1, rng=None):
        abb = self.abb
        u = self.uov
        g = self.gf
        rng = rng or abb.rng
        ok = True
        for _ in range(reps):
            r = g.rand(rng, (u.m,))
            c = g.rand(rng, (u.o,))
            Q = np.zeros((u.v, u.v), dtype=g.dtype)
            R = np.zeros((u.v, u.o), dtype=g.dtype)
            z = np.zeros((1, 1), dtype=g.dtype)
            for i in range(u.m):
                Q ^= g.mul(r[i], P1[i])
                R ^= g.mul(r[i], P2[i])
                z ^= g.mul(r[i], g.matmul(c.reshape(1, -1), g.matmul(P3[i], c.reshape(-1, 1))))
            su = abb.matmul_public_right(sO, c.reshape(-1, 1)).reshape(u.v)
            Qu = abb.matmul_public_left(Q, su.reshape(u.v, 1)).reshape(u.v)
            quad = abb.mul_begin(su.reshape(1, u.v), Qu.reshape(u.v, 1), "mm")
            abb.flush()
            lin = abb.matmul_public_left((g.matmul(R, c.reshape(-1, 1))).reshape(1, -1),
                                         su.reshape(u.v, 1))
            total = quad.out + lin
            val = abb.open(total)
            ok = ok and int(val.reshape(-1)[0]) == int(z[0, 0])
        return ok

    def key_material(self, P1, P2, sO):
        abb = self.abb
        u = self.uov
        g = self.gf
        mats = []
        for i in range(u.m):
            S = abb.matmul_public_left(P1[i] ^ P1[i].T, sO)
            mats.append(abb.add_public(S, P2[i]))
        masked = MaskedMatrix.batch(abb, [mm.transpose() for mm in mats])
        return {"M": mats, "masked": masked, "P1": P1, "P2": P2, "sO": sO}

    def offline(self, km, block=None, max_trials=6):
        abb = self.abb
        u = self.uov
        g = self.gf
        sv = abb.rand((u.v,))
        rows = run_ops(abb, [(tuple(km["masked"]), sv)])[0]
        sA = Shared(abb, np.stack([r.shares for r in rows], axis=1))
        sw = quadratic_forms(abb, sv, km["P1"])

        res = ours.offline_online_split(abb, sA, u.m, u.o, block=block, max_trials=max_trials)
        if not res["ok"]:
            return None
        Gamma = res["Gamma"]
        cker = res["c"]
        Gw = abb.matvec(Gamma, sw)
        c = cker + Gw
        OG = abb.matmul(km["sO"], Gamma)
        Oc = abb.matvec(km["sO"], c)
        return {"v": sv, "Gamma": Gamma, "c": c, "OG": OG, "Oc": Oc}

    def online(self, pre, msg, salt):
        abb = self.abb
        u = self.uov
        g = self.gf
        t = u.hash_target(msg, salt)
        tv = t.reshape(-1, 1)
        x = abb.matmul_public_right(pre["Gamma"], tv).reshape(u.o) + pre["c"]
        s1 = pre["v"] + abb.matmul_public_right(pre["OG"], tv).reshape(u.v) + pre["Oc"]
        sig = Shared(abb, np.concatenate([s1.shares, x.shares], axis=1))
        return abb.open(sig)
