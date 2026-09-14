import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsov.gf import GF
from tsov.robust import robust_open, shamir_share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/robust.csv")
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    gf = GF(8)
    rng = np.random.default_rng(17)
    rows = []
    for (n, t) in ((5, 1), (7, 2), (10, 3), (13, 4), (16, 5)):
        emax = (n - t - 1) // 2
        pts = [gf.dtype(i + 1) for i in range(n)]
        for e in range(emax + 1):
            good = 0
            ident = 0
            t0 = time.perf_counter()
            for _ in range(args.reps):
                val = gf.rand(rng, (32,))
                sh = shamir_share(gf, val, n, t, pts, rng)
                cheats = list(rng.permutation(n)[:e])
                for c in cheats:
                    sh[c] ^= gf.rand_nonzero(rng, (32,))
                out, bad = robust_open(gf, pts, sh, t)
                if out is not None and np.array_equal(out, val):
                    good += 1
                    if sorted(bad) == sorted(int(c) for c in cheats):
                        ident += 1
            el = (time.perf_counter() - t0) / args.reps
            rows.append({"n": n, "t": t, "errors": e, "max_errors": emax,
                         "recovered": good / float(args.reps),
                         "identified": ident / float(args.reps), "time": el})
            print("n=%2d t=%d e=%d recovered=%.2f identified=%.2f  %.2f ms"
                  % (n, t, e, rows[-1]["recovered"], rows[-1]["identified"], el * 1000))
            sys.stdout.flush()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("written", args.out)


if __name__ == "__main__":
    main()
