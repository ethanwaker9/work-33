import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsov.abb import ABB
from tsov.gf import GF
from tsov.solvers import ours


def condense_plain(gf, B, s, b):
    D = np.array(B, dtype=gf.dtype)
    while s > b:
        bb = min(b, s - 1)
        B11 = D[:bb, :bb]
        B12 = D[:bb, bb:]
        B21 = D[bb:, :bb]
        B22 = D[bb:, bb:]
        d = gf.det(B11)
        adj = gf.inv_matrix(B11)
        if adj is None:
            return 0
        adj = gf.mul(np.array(int(d), dtype=gf.dtype), adj)
        D = gf.mul(np.array(int(d), dtype=gf.dtype), B22) ^ gf.matmul(B21, gf.matmul(adj, B12))
        s -= bb
    return int(gf.det(D))


def success_rate(gf, s, b, trials, rng):
    good = 0
    for _ in range(trials):
        B = gf.rand(rng, (s, s))
        if gf.det(B) == 0:
            continue
        good += 1 if condense_plain(gf, B, s, b) != 0 else 0
    return good / float(trials)


def predicted(gf, s, b):
    q = gf.q
    alpha = 1.0
    for i in range(1, b + 1):
        alpha *= 1.0 - q ** (-i)
    steps = 0
    ss = s
    while ss > b:
        ss -= min(b, ss - 1)
        steps += 1
    tail = 1.0
    for i in range(1, ss + 1):
        tail *= 1.0 - q ** (-i)
    return (alpha ** steps) * tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/blocksize.csv")
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()
    rows = []
    for deg in (4, 8):
        gf = GF(deg)
        rng = np.random.default_rng(7 + deg)
        for s in (24, 32, 44, 64):
            for b in (1, 2, 3, 4):
                emp = success_rate(gf, s, b, args.trials, rng)
                rows.append({"q": gf.q, "s": s, "b": b,
                             "empirical": emp, "predicted": predicted(gf, s, b)})
                print("q=%3d s=%3d b=%d empirical=%.3f predicted=%.3f"
                      % (gf.q, s, b, emp, rows[-1]["predicted"]))
                sys.stdout.flush()

    cost = []
    gf = GF(8)
    for s in (32, 48, 64):
        for b in (1, 2, 3, 4, 6, 8):
            abb = ABB(gf, 3, 5 + b)
            rng = np.random.default_rng(3 * s + b)
            A = gf.rand(rng, (s, s))
            y = gf.rand(rng, (s,))
            sA, sy = abb.share(A), abb.share(y)
            t0 = time.perf_counter()
            res = ours.solve(abb, sA, sy, s, s, block=b, max_trials=12)
            el = time.perf_counter() - t0 - abb.prep_time
            cost.append({"s": s, "b": b, "rounds": abb.rounds, "comm": abb.comm,
                         "time": el, "trials": res["trials"], "ok": res["ok"]})
            print("cost s=%3d b=%d rounds=%4d comm=%8d time=%.4f trials=%d"
                  % (s, b, abb.rounds, abb.comm, el, res["trials"]))
            sys.stdout.flush()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(args.out.replace(".csv", "_cost.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cost[0].keys()))
        w.writeheader()
        for r in cost:
            w.writerow(r)
    print("written", args.out)


if __name__ == "__main__":
    main()
