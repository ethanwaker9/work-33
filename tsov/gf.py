import numpy as np

_PRIM = {2: 0x7, 3: 0xB, 4: 0x13, 5: 0x25, 6: 0x43, 7: 0x83, 8: 0x11D}


class GF:
    _cache = {}

    def __new__(cls, deg):
        if deg in cls._cache:
            return cls._cache[deg]
        obj = super().__new__(cls)
        obj._init(deg)
        cls._cache[deg] = obj
        return obj

    def _init(self, deg):
        if deg not in _PRIM:
            raise ValueError("unsupported extension degree")
        self.deg = deg
        self.q = 1 << deg
        self.dtype = np.uint8 if deg <= 8 else np.uint16
        poly = _PRIM[deg]
        exp = np.zeros(2 * self.q, dtype=self.dtype)
        log = np.zeros(self.q, dtype=np.int32)
        x = 1
        for i in range(self.q - 1):
            exp[i] = x
            log[x] = i
            x <<= 1
            if x & self.q:
                x ^= poly
        for i in range(self.q - 1, 2 * self.q):
            exp[i] = exp[i - (self.q - 1)]
        if len(set(int(v) for v in exp[: self.q - 1])) != self.q - 1:
            raise RuntimeError("generator is not primitive")
        log[0] = -1
        self.exp = exp
        self.log = log
        big = 4 * self.q
        expx = np.zeros(2 * big + 2, dtype=self.dtype)
        expx[: 2 * self.q] = exp
        self.expx = expx
        self.logz = big
        self.logpos = np.where(log < 0, big, log).astype(np.int16)
        table = np.zeros((self.q, self.q), dtype=self.dtype)
        idx = np.arange(1, self.q)
        la = log[idx]
        s = (la[:, None] + la[None, :]) % (self.q - 1)
        table[1:, 1:] = exp[s]
        self.mul_table = table
        inv = np.zeros(self.q, dtype=self.dtype)
        inv[1:] = exp[(self.q - 1 - log[idx]) % (self.q - 1)]
        self.inv_table = inv

    def zeros(self, shape):
        return np.zeros(shape, dtype=self.dtype)

    def rand(self, rng, shape):
        return rng.integers(0, self.q, size=shape, dtype=np.uint64).astype(self.dtype)

    def rand_nonzero(self, rng, shape):
        return rng.integers(1, self.q, size=shape, dtype=np.uint64).astype(self.dtype)

    def add(self, a, b):
        return np.bitwise_xor(a, b)

    def mul(self, a, b):
        a = np.asarray(a, dtype=self.dtype)
        b = np.asarray(b, dtype=self.dtype)
        a, b = np.broadcast_arrays(a, b)
        return self.mul_table[a, b]

    def inv(self, a):
        return self.inv_table[np.asarray(a, dtype=self.dtype)]

    def matmul(self, A, B):
        A = np.atleast_2d(np.asarray(A, dtype=self.dtype))
        B = np.atleast_2d(np.asarray(B, dtype=self.dtype))
        n, k = A.shape
        k2, p = B.shape
        if k != k2:
            raise ValueError("shape mismatch")
        if n * p >= 256 and k >= 4:
            chunk = max(1, min(k, 262144 // max(1, n * p)))
            C = np.zeros((n, p), dtype=self.dtype)
            for j0 in range(0, k, chunk):
                j1 = min(k, j0 + chunk)
                Ab = A[:, j0:j1]
                Bb = B[j0:j1, :]
                ssum = self.logpos[Ab][:, :, None] + self.logpos[Bb][None, :, :]
                C ^= np.bitwise_xor.reduce(self.expx[ssum], axis=1)
            return C
        C = np.zeros((n, p), dtype=self.dtype)
        for j in range(k):
            col = A[:, j]
            row = B[j, :]
            nz = col != 0
            if not nz.any():
                continue
            C[nz] ^= self.mul_table[col[nz][:, None], row[None, :]]
        return C

    def matvec(self, A, v):
        return self.matmul(A, np.asarray(v, dtype=self.dtype).reshape(-1, 1)).reshape(-1)

    def rref(self, M):
        M = np.array(M, dtype=self.dtype)
        rows, cols = M.shape
        pivots = []
        r = 0
        for c in range(cols):
            piv = None
            for i in range(r, rows):
                if M[i, c]:
                    piv = i
                    break
            if piv is None:
                continue
            if piv != r:
                M[[r, piv]] = M[[piv, r]]
            iv = self.inv_table[M[r, c]]
            M[r] = self.mul_table[M[r], iv]
            col = M[:, c].copy()
            col[r] = 0
            nz = col != 0
            if nz.any():
                M[nz] ^= self.mul_table[col[nz][:, None], M[r][None, :]]
            pivots.append(c)
            r += 1
            if r == rows:
                break
        return M, pivots

    def rank(self, M):
        _, piv = self.rref(M)
        return len(piv)

    def inv_matrix(self, M):
        M = np.asarray(M, dtype=self.dtype)
        n = M.shape[0]
        aug = np.concatenate([M, np.eye(n, dtype=self.dtype)], axis=1)
        R, piv = self.rref(aug)
        if len(piv) < n or piv[-1] >= n:
            return None
        return R[:, n:]

    def det(self, M):
        M = np.array(M, dtype=self.dtype)
        n = M.shape[0]
        d = self.dtype(1)
        for c in range(n):
            piv = None
            for i in range(c, n):
                if M[i, c]:
                    piv = i
                    break
            if piv is None:
                return self.dtype(0)
            if piv != c:
                M[[c, piv]] = M[[piv, c]]
            d = self.mul_table[d, M[c, c]]
            iv = self.inv_table[M[c, c]]
            M[c] = self.mul_table[M[c], iv]
            col = M[c + 1:, c]
            nz = col != 0
            if nz.any():
                idx = np.arange(c + 1, n)[nz]
                M[idx] ^= self.mul_table[col[nz][:, None], M[c][None, :]]
        return d

    def solve(self, A, b):
        A = np.asarray(A, dtype=self.dtype)
        b = np.asarray(b, dtype=self.dtype).reshape(-1, 1)
        aug = np.concatenate([A, b], axis=1)
        R, piv = self.rref(aug)
        cols = A.shape[1]
        if cols in piv:
            return None
        x = np.zeros(cols, dtype=self.dtype)
        for i, c in enumerate(piv):
            x[c] = R[i, cols]
        return x

    def right_inverse(self, B):
        return self.inv_matrix(B)

    def subfield_embedding(self, sub):
        if self.deg % sub.deg != 0:
            raise ValueError("not a subfield")
        poly = _PRIM[sub.deg]
        beta = None
        for cand in range(1, self.q):
            acc = self.dtype(1)
            val = self.dtype(0)
            c = self.dtype(cand)
            for bit in range(sub.deg + 1):
                if (poly >> bit) & 1:
                    val ^= acc
                acc = self.mul_table[acc, c]
            if val == 0 and cand != 0:
                beta = c
                break
        if beta is None:
            raise RuntimeError("no root found")
        up = np.zeros(sub.q, dtype=self.dtype)
        for x in range(sub.q):
            acc = self.dtype(1)
            out = self.dtype(0)
            for bit in range(sub.deg):
                if (x >> bit) & 1:
                    out ^= acc
                acc = self.mul_table[acc, beta]
            up[x] = out
        down = np.full(self.q, 255, dtype=np.int16)
        for x in range(sub.q):
            down[int(up[x])] = x
        return up, down
