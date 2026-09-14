import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsov.abb import ABB
from tsov.gf import GF
from tsov.robust import robust_open, shamir_share
from tsov.solvers import aobv, masking, ours
from tsov.solvers.common import berkowitz_det, det_adj
from tsov.solvers.newton import newton_det
from tsov.threshold import ThresholdUOV
from tsov.uov import UOV

PASS = []


def check(name, cond):
    PASS.append((name, bool(cond)))
    print("%-46s %s" % (name, "ok" if cond else "FAILED"))


def test_field():
    for deg in (4, 8):
        g = GF(deg)
        rng = np.random.default_rng(1)
        A = g.rand(rng, (9, 7))
        B = g.rand(rng, (7, 6))
        ref = np.zeros((9, 6), dtype=g.dtype)
        for k in range(7):
            ref ^= g.mul_table[A[:, k][:, None], B[k, :][None, :]]
        check("GF(2^%d) matrix product" % deg, np.array_equal(g.matmul(A, B), ref))
        M = g.rand(rng, (8, 8))
        Mi = g.inv_matrix(M)
        if Mi is not None:
            check("GF(2^%d) matrix inverse" % deg,
                  np.array_equal(g.matmul(M, Mi), np.eye(8, dtype=g.dtype)))
        x = g.rand(rng, (8,))
        y = g.matvec(M, x)
        sol = g.solve(M, y)
        check("GF(2^%d) linear solve" % deg,
              sol is not None and np.array_equal(g.matvec(M, sol), y))


def test_subfield():
    g8, g4 = GF(8), GF(4)
    up, down = g8.subfield_embedding(g4)
    ok = True
    for a in range(16):
        for b in range(16):
            ok &= int(g8.mul_table[up[a], up[b]]) == int(up[int(g4.mul_table[a, b])])
            ok &= int(up[a] ^ up[b]) == int(up[a ^ b])
    check("subfield embedding is a homomorphism", ok)
    check("subfield inverse map", all(down[int(up[x])] == x for x in range(16)))


def test_abb():
    g = GF(8)
    abb = ABB(g, 3, 5)
    rng = np.random.default_rng(2)
    X, Y = g.rand(rng, (6, 5)), g.rand(rng, (5, 4))
    check("Beaver matrix product",
          np.array_equal(abb.matmul(abb.share(X), abb.share(Y)).value, g.matmul(X, Y)))
    v = g.rand(rng, (5,))
    check("Beaver matrix-vector product",
          np.array_equal(abb.matvec(abb.share(X), abb.share(v)).value, g.matvec(X, v)))
    s = np.array(np.uint8(11), dtype=g.dtype)
    check("Beaver scalar-matrix product",
          np.array_equal(abb.scalmul(abb.share(s), abb.share(X)).value, g.mul(s, X)))


def test_determinants():
    g = GF(8)
    rng = np.random.default_rng(3)
    for n in (2, 3, 5, 9):
        abb = ABB(g, 3, n)
        M = g.rand(rng, (n, n))
        check("Berkowitz determinant n=%d" % n,
              int(berkowitz_det(abb, abb.share(M), n).value) == int(g.det(M)))
    for b in (1, 2, 3, 4, 6):
        abb = ABB(g, 3, b)
        M = g.rand(rng, (b, b))
        d, adj = det_adj(abb, abb.share(M), b)
        want = g.mul(np.array(int(g.det(M)), dtype=g.dtype), np.eye(b, dtype=g.dtype))
        check("det and adjugate b=%d" % b,
              int(d.value) == int(g.det(M)) and np.array_equal(g.matmul(M, adj.value), want))
    for n in (4, 6):
        for mode in ("bsgs", "all"):
            abb = ABB(g, 3, n)
            M = g.rand(rng, (n, n))
            check("generalized Newton determinant n=%d %s" % (n, mode),
                  int(newton_det(abb, abb.share(M), n, mode).value) == int(g.det(M)))


def test_solvers():
    g = GF(8)
    rng = np.random.default_rng(4)
    fns = [("Cozzo-Smart", masking.solve_cs), ("Celi et al.", masking.solve_cen),
           ("Berkowitz", aobv.solve_berkowitz), ("Newton", aobv.solve_newton),
           ("Cramer-Damgard", aobv.solve_cd01), ("BDFC", ours.solve)]
    for (s, t) in ((7, 7), (7, 10)):
        for name, fn in fns:
            abb = ABB(g, 3, s + 1)
            A, y = g.rand(rng, (s, t)), g.rand(rng, (s,))
            res = fn(abb, abb.share(A), abb.share(y), s, t)
            ok = res["ok"] and np.array_equal(g.matvec(A, res["x"].value), y)
            check("solver %s (%dx%d)" % (name, s, t), ok)


def test_solver_subfield():
    g8, g4 = GF(8), GF(4)
    up, down = g8.subfield_embedding(g4)
    rng = np.random.default_rng(6)
    s, t = 16, 20
    abb = ABB(g8, 3, 9)
    A, y = g4.rand(rng, (s, t)), g4.rand(rng, (s,))
    res = ours.solve_subfield(abb, abb.share(up[A]), abb.share(up[y]), s, t, up, block=4)
    ok = False
    if res["ok"]:
        xs = down[res["x"].value]
        ok = bool((xs >= 0).all()) and np.array_equal(
            g4.matvec(A, xs.astype(g4.dtype)), y)
    check("BDFC over subfield embedding", ok)


def test_onesided():
    g = GF(8)
    rng = np.random.default_rng(7)
    bad = 0
    for i in range(40):
        abb = ABB(g, 3, 40 + i)
        M = g.rand(rng, (12, 12))
        M[3] = np.bitwise_xor.reduce(M[[0, 1, 2]], axis=0)
        d = ours.block_condense_det(abb, abb.share(M), 12, 4)
        if int(d.value) != 0:
            bad += 1
    check("no false accept on singular matrices", bad == 0)


def test_uov():
    for name in ("TEST", "UOV-Ip"):
        u = UOV(name, seed=11)
        pk, O = u.keygen()
        sig = u.sign(pk, O, b"unit test")
        check("UOV %s sign and verify" % name, u.verify(pk, b"unit test", sig))
        check("UOV %s rejects other message" % name,
              not u.verify(pk, b"other message", sig))


def test_threshold():
    u = UOV("TEST", seed=13)
    abb = ABB(u.gf, 3, 17)
    th = ThresholdUOV(abb, u)
    P1, P2 = u.public_seed_material()
    sO, P3 = th.dkg(P1, P2)
    check("DKG matches local key derivation",
          np.array_equal(P3, u.derive_p3(P1, P2, sO.value)))
    check("batched public key check accepts", th.verify_key(P1, P2, P3, sO, reps=2))
    tampered = P3.copy()
    tampered[0, 0, 0] ^= 1
    check("batched public key check rejects",
          not th.verify_key(P1, P2, tampered, sO, reps=6))
    km = th.key_material(P1, P2, sO)
    good = True
    for i in range(3):
        pre = th.offline(km)
        salt = bytes([i] * 24)
        sig = th.online(pre, b"threshold message", salt)
        good = good and u.verify((P1, P2, P3), b"threshold message", (sig, salt))
    check("threshold signatures verify", good)


def test_robust():
    g = GF(8)
    rng = np.random.default_rng(19)
    for (n, t) in ((7, 2), (13, 4)):
        pts = [g.dtype(i + 1) for i in range(n)]
        emax = (n - t - 1) // 2
        val = g.rand(rng, (16,))
        sh = shamir_share(g, val, n, t, pts, rng)
        cheats = sorted(int(c) for c in rng.permutation(n)[:emax])
        for c in cheats:
            sh[c] ^= g.rand_nonzero(rng, (16,))
        out, bad = robust_open(g, pts, sh, t)
        check("robust opening N=%d tau=%d e=%d" % (n, t, emax),
              out is not None and np.array_equal(out, val) and bad == cheats)


def main():
    test_field()
    test_subfield()
    test_abb()
    test_determinants()
    test_solvers()
    test_solver_subfield()
    test_onesided()
    test_uov()
    test_threshold()
    test_robust()
    failed = [n for n, ok in PASS if not ok]
    print("\n%d checks, %d failed" % (len(PASS), len(failed)))
    for n in failed:
        print("  FAILED:", n)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
