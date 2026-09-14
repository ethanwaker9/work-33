import numpy as np

CS_NAME = "Cozzo-Smart"
CEN_NAME = "Celi-Escudero-Niot"


def _kernel_basis(gf, T, t):
    R, piv = gf.rref(T)
    free = [c for c in range(t) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(t, dtype=gf.dtype)
        v[f] = 1
        for i, c in enumerate(piv):
            v[c] = R[i, f]
        basis.append(v)
    if not basis:
        return np.zeros((t, 0), dtype=gf.dtype)
    return np.stack(basis, axis=1)


def solve_cs(abb, A, y, s, t, max_trials=3):
    g = abb.gf
    trials = 0
    for _ in range(max_trials):
        trials += 1
        if t > s:
            S = abb.rand((t, s))
            A2 = abb.matmul(A, S)
        else:
            S = None
            A2 = A
        R = abb.rand((s, s))
        T = abb.matmul(R, A2)
        Tv = abb.open(T)
        Tinv = g.inv_matrix(Tv)
        if Tinv is None:
            continue
        Ry = abb.matvec(R, y)
        z = abb.matmul_public_left(Tinv, Ry.reshape(s, 1)).reshape(s)
        x = abb.matvec(S, z) if S is not None else z
        return {"ok": True, "x": x, "trials": trials, "leak": "image and kernel"}
    return {"ok": False, "x": None, "trials": trials, "leak": "image and kernel"}


def solve_cen(abb, A, y, s, t, max_trials=3):
    g = abb.gf
    trials = 0
    for _ in range(max_trials):
        trials += 1
        R = abb.rand((s, s))
        S = abb.rand((t, t))
        AS = abb.matmul(A, S)
        T = abb.matmul(R, AS)
        Tv = abb.open(T)
        if g.rank(Tv) < s:
            continue
        Tri = _right_inverse(g, Tv, s, t)
        SB = abb.matmul_public_right(S, Tri)
        Ainv = abb.matmul(SB, R)
        x = abb.matvec(Ainv, y)
        if t > s:
            K = _kernel_basis(g, Tv, t)
            zz = abb.rand((K.shape[1],))
            z = abb.matmul_public_left(K, zz.reshape(K.shape[1], 1)).reshape(t)
            x = x + abb.matvec(S, z)
        return {"ok": True, "x": x, "trials": trials, "leak": "rank"}
    return {"ok": False, "x": None, "trials": trials, "leak": "rank"}


def _right_inverse(gf, T, s, t):
    if s == t:
        return gf.inv_matrix(T)
    aug = np.concatenate([T, np.eye(s, dtype=gf.dtype)], axis=1)
    R, piv = gf.rref(aug)
    out = np.zeros((t, s), dtype=gf.dtype)
    for i, c in enumerate(piv):
        if c >= t:
            return None
        out[c] = R[i, t:]
    return out
