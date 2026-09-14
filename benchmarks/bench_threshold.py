import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsov.abb import ABB
from tsov.threshold import ThresholdUOV
from tsov.uov import UOV


def run(name, parties, reps, seed=11):
    u = UOV(name, seed=seed)
    abb = ABB(u.gf, parties, seed)
    th = ThresholdUOV(abb, u)
    P1, P2 = u.public_seed_material()

    c0, r0, t0 = abb.comm, abb.rounds, time.perf_counter()
    sO, P3 = th.dkg(P1, P2)
    dkg_time = time.perf_counter() - t0 - abb.prep_time
    dkg_comm, dkg_rounds = abb.comm - c0, abb.rounds - r0
    pt = abb.prep_time

    ok_key = np.array_equal(P3, u.derive_p3(P1, P2, sO.value))
    c0, r0, t0 = abb.comm, abb.rounds, time.perf_counter()
    ok_ver = th.verify_key(P1, P2, P3, sO, reps=2)
    ver_time = time.perf_counter() - t0 - (abb.prep_time - pt)
    ver_comm, ver_rounds = abb.comm - c0, abb.rounds - r0

    km = th.key_material(P1, P2, sO)

    off_comm = off_rounds = on_comm = on_rounds = 0
    off_time = on_time = 0.0
    all_valid = True
    for i in range(reps):
        pt = abb.prep_time
        c0, r0, t0 = abb.comm, abb.rounds, time.perf_counter()
        pre = th.offline(km)
        off_time += (time.perf_counter() - t0 - (abb.prep_time - pt)) / reps
        off_comm += (abb.comm - c0) / reps
        off_rounds += (abb.rounds - r0) / reps
        salt = bytes([i] * 24)
        pt = abb.prep_time
        c0, r0, t0 = abb.comm, abb.rounds, time.perf_counter()
        sig = th.online(pre, b"benchmark message", salt)
        on_time += (time.perf_counter() - t0 - (abb.prep_time - pt)) / reps
        on_comm += (abb.comm - c0) / reps
        on_rounds += (abb.rounds - r0) / reps
        all_valid = all_valid and u.verify((P1, P2, P3), b"benchmark message", (sig, salt))

    return {
        "scheme": name, "n": u.n, "m": u.m, "o": u.o, "q": u.gf.q, "parties": parties,
        "dkg_comm": dkg_comm, "dkg_rounds": dkg_rounds, "dkg_time": dkg_time,
        "dkg_correct": ok_key, "keycheck_ok": ok_ver, "keycheck_comm": ver_comm,
        "keycheck_rounds": ver_rounds, "keycheck_time": ver_time,
        "off_comm": off_comm, "off_rounds": off_rounds, "off_time": off_time,
        "on_comm": on_comm, "on_rounds": on_rounds, "on_time": on_time,
        "sig_valid": all_valid,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/threshold.csv")
    ap.add_argument("--parties", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    rows = []
    for name in ("UOV-Ip", "UOV-III", "UOV-V"):
        row = run(name, args.parties, args.reps)
        rows.append(row)
        print("%-8s dkg %.2fs/%d elts | offline %.3fs/%.0f elts/%.0f rds | online %.4fs/%.0f elts/%.0f rds | valid=%s"
              % (name, row["dkg_time"], row["dkg_comm"], row["off_time"], row["off_comm"],
                 row["off_rounds"], row["on_time"], row["on_comm"], row["on_rounds"],
                 row["sig_valid"]))
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
