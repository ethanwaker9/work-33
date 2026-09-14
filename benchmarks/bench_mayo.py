import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsov.abb import ABB
from tsov.gf import GF
from tsov.solvers import aobv, masking, ours

SHAPES = [("MAYO-1", 64, 72), ("MAYO-3", 96, 110), ("MAYO-5", 128, 144)]


def summarize(name, method, s, t, reps, runner, bits):
    agg = {"rounds": 0.0, "comm_bytes": 0.0, "time": 0.0, "trials": 0.0, "mem_bytes": 0.0}
    ok = True
    for r in range(reps):
        stats, good = runner(r)
        ok = ok and good
        agg["rounds"] += stats["rounds"] / reps
        agg["comm_bytes"] += stats["comm"] * bits / 8.0 / reps
        agg["time"] += stats["time"] / reps
        agg["trials"] += stats["trials"] / reps
        agg["mem_bytes"] += stats["peak"] * bits / 8.0 / reps
    agg.update({"scheme": name, "method": method, "s": s, "t": t, "ok": ok})
    return agg


def native_runner(gf, fn, s, t, seed, block=None):
    def run(r):
        abb = ABB(gf, 3, seed + r)
        rng = np.random.default_rng(seed + 31 * r)
        A = gf.rand(rng, (s, t))
        y = gf.rand(rng, (s,))
        sA, sy = abb.share(A), abb.share(y)
        t0 = time.perf_counter()
        kw = {"max_trials": 30}
        if block is not None:
            kw["block"] = block
        res = fn(abb, sA, sy, s, t, **kw)
        el = time.perf_counter() - t0 - abb.prep_time
        good = res["ok"] and np.array_equal(gf.matvec(A, res["x"].value), y)
        return ({"rounds": abb.rounds, "comm": abb.comm, "time": el,
                 "trials": res["trials"], "peak": abb.peak_bytes}, good)
    return run


def ext_runner(gsub, gbig, up, down, s, t, seed, block=4):
    def run(r):
        abb = ABB(gbig, 3, seed + r)
        rng = np.random.default_rng(seed + 31 * r)
        A = gsub.rand(rng, (s, t))
        y = gsub.rand(rng, (s,))
        sA, sy = abb.share(up[A]), abb.share(up[y])
        t0 = time.perf_counter()
        res = ours.solve_subfield(abb, sA, sy, s, t, up, block=block, max_trials=6)
        el = time.perf_counter() - t0 - abb.prep_time
        good = False
        if res["ok"]:
            x8 = res["x"].value
            xs = down[x8]
            good = bool((xs >= 0).all()) and np.array_equal(
                gsub.matvec(A, xs.astype(gsub.dtype)), y)
        return ({"rounds": abb.rounds, "comm": abb.comm, "time": el,
                 "trials": res["trials"], "peak": abb.peak_bytes}, good)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/mayo.csv")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    g4, g8 = GF(4), GF(8)
    up, down = g8.subfield_embedding(g4)
    rows = []
    for name, s, t in SHAPES:
        rows.append(summarize(name, "Celi et al.", s, t, args.reps,
                              native_runner(g4, masking.solve_cen, s, t, 501), 4))
        rows.append(summarize(name, "Berkowitz", s, t, 1,
                              native_runner(g4, aobv.solve_berkowitz, s, t, 502), 4))
        rows.append(summarize(name, "BDFC native", s, t, args.reps,
                              native_runner(g4, ours.solve, s, t, 503, block=4), 4))
        rows.append(summarize(name, "BDFC extended", s, t, args.reps,
                              ext_runner(g4, g8, up, down, s, t, 504), 8))
        for r in rows[-4:]:
            print("%-7s %-14s rounds=%6.0f comm=%9.0f B time=%7.3f trials=%4.1f mem=%9.0f B ok=%s"
                  % (name, r["method"], r["rounds"], r["comm_bytes"], r["time"],
                     r["trials"], r["mem_bytes"], r["ok"]))
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
