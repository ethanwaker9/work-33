import argparse
import csv
import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsov.abb import ABB
from tsov.gf import GF
from tsov.solvers import aobv, masking, ours

METHODS = {
    "CS19": (masking.solve_cs, "image and kernel"),
    "CEN25": (masking.solve_cen, "rank"),
    "CD01": (aobv.solve_cd01, "none"),
    "CKP07": (aobv.solve_newton, "none"),
    "AOBV-B": (aobv.solve_berkowitz, "none"),
    "BDFC": (ours.solve, "none"),
}

FAST = ["CS19", "CEN25", "AOBV-B", "BDFC"]


def one_run(gf, method, s, t, parties, seed, measure_mem=False):
    fn = METHODS[method][0]
    abb = ABB(gf, parties, seed)
    rng = np.random.default_rng(seed + 977)
    A = gf.rand(rng, (s, t))
    y = gf.rand(rng, (s,))
    sA = abb.share(A)
    sy = abb.share(y)
    if measure_mem:
        tracemalloc.start()
        tracemalloc.reset_peak()
    t0 = time.perf_counter()
    res = fn(abb, sA, sy, s, t)
    elapsed = time.perf_counter() - t0 - abb.prep_time
    peak = 0
    if measure_mem:
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    ok = False
    if res["ok"]:
        ok = np.array_equal(gf.matvec(A, res["x"].value), y)
    return {
        "method": method,
        "s": s,
        "t": t,
        "parties": parties,
        "ok": ok,
        "trials": res["trials"],
        "rounds": abb.rounds,
        "comm": abb.comm,
        "prep": abb.prep,
        "mults": abb.mults,
        "time": elapsed,
        "peak_bytes": peak,
        "peak_shared": abb.peak_bytes,
        "leakage": METHODS[method][1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/solvers.csv")
    ap.add_argument("--parties", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--deg", type=int, default=8)
    args = ap.parse_args()

    gf = GF(args.deg)
    plan = []
    for s in (8, 12, 16, 20, 24, 32):
        for m in METHODS:
            reps = args.reps if (m in FAST or s <= 16) else 1
            plan.append((m, s, s, reps))
    for s in (44, 48, 64, 72, 96):
        for m in FAST:
            plan.append((m, s, s, max(2, args.reps - 1)))
    plan.append(("CKP07", 44, 44, 1))
    plan.append(("CD01", 44, 44, 1))

    rows = []
    for method, s, t, reps in plan:
        agg = {"rounds": 0.0, "comm": 0.0, "prep": 0.0, "mults": 0.0,
               "trials": 0.0, "time": 0.0, "peak_shared": 0.0}
        ok = True
        for r in range(reps):
            out = one_run(gf, method, s, t, args.parties, 1000 + 7 * r + s)
            for key in agg:
                agg[key] += out[key] / reps
            ok = ok and out["ok"]
        row = {"method": method, "s": s, "t": t, "parties": args.parties, "ok": ok,
               "leakage": METHODS[method][1]}
        row.update(agg)
        rows.append(row)
        print("%-8s s=%3d rounds=%7.1f comm=%10.0f time=%8.4fs mem=%8.1fKB ok=%s"
              % (method, s, row["rounds"], row["comm"], row["time"],
                 row["peak_shared"] / 1024.0, ok))
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
