import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

SC = 2.0
ORDER = ["CS19", "CEN25", "CD01", "CKP07", "AOBV-B", "BDFC"]
PLAIN = {
    "CS19": "Cozzo-Smart (leaks kernel)",
    "CEN25": "Celi et al. (leaks rank)",
    "CD01": "Cramer-Damgard, lifted",
    "CKP07": "Newton, lifted",
    "AOBV-B": "Berkowitz",
    "BDFC": "BDFC (this work)",
}
STYLE = {
    "CS19": ("o", ":", "#7f7f7f"),
    "CEN25": ("s", ":", "#8c564b"),
    "CD01": ("^", "--", "#2ca02c"),
    "CKP07": ("v", "--", "#9467bd"),
    "AOBV-B": ("D", "-.", "#1f77b4"),
    "BDFC": ("*", "-", "#d62728"),
}


def load(path):
    with open(path) as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def series(rows, method, key):
    pts = sorted([(int(r["s"]), float(r[key])) for r in rows if r["method"] == method])
    return [p[0] for p in pts], [p[1] for p in pts]


def save(fig, out, name):
    eps = os.path.join(out, name + ".eps")
    fig.savefig(eps, format="eps", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    os.system("epstopdf --outfile=%s %s" % (os.path.join(out, name + ".pdf"), eps))
    print("wrote", name)


def fig_scaling(rows, out, blk=None):
    keys = [("comm", "communication (elements/party)"),
            ("time", "local computation time (s)"),
            ("peak_shared", "peak shared state (KB/party)"),
            ("rounds", "interaction rounds")]
    ncol = 5 if blk else 4
    fig, axes = plt.subplots(1, ncol, figsize=(7.1 * SC, 1.42 * SC))
    for ax, (key, ylab) in zip(axes[:4], keys):
        for m in ORDER:
            xs, ys = series(rows, m, key)
            if not xs:
                continue
            if key == "peak_shared":
                ys = [y / (1024.0 * 3) for y in ys]
            mk, ls, col = STYLE[m]
            ax.plot(xs, ys, marker=mk, linestyle=ls, color=col, label=PLAIN[m],
                    markersize=3.2 * SC, linewidth=0.8 * SC)
        ax.set_xscale("log", base=2)
        if key != "rounds":
            ax.set_yscale("log")
        ax.set_xlabel("system size $m$", fontsize=6.2 * SC)
        ax.set_ylabel(ylab, fontsize=6.2 * SC)
        ax.grid(True, which="both", alpha=0.3, linewidth=0.3 * SC)
        ax.tick_params(labelsize=5.6 * SC, width=0.4 * SC, length=1.6 * SC)
    axes[0].legend(fontsize=4.4 * SC, loc="upper left", framealpha=0.92,
                   handlelength=1.4, borderpad=0.25, labelspacing=0.2)
    if blk:
        ax = axes[4]
        for q, col in ((16, "#d62728"), (256, "#1f77b4")):
            for sz, mk in ((32, "o"), (64, "s")):
                sub = sorted([(int(r["b"]), float(r["empirical"]), float(r["predicted"]))
                              for r in blk if int(r["q"]) == q and int(r["s"]) == sz])
                if not sub:
                    continue
                ax.plot([x[0] for x in sub], [x[1] for x in sub], marker=mk,
                        linestyle="none", color=col, markersize=3.0 * SC,
                        label="$q{=}%d$, $m{=}%d$" % (q, sz))
                ax.plot([x[0] for x in sub], [x[2] for x in sub], linestyle="--",
                        color=col, linewidth=0.7 * SC)
        ax.set_xlabel("block size $b$", fontsize=6.2 * SC)
        ax.set_ylabel("per-attempt success prob.", fontsize=6.2 * SC)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, which="both", alpha=0.3, linewidth=0.3 * SC)
        ax.legend(fontsize=4.4 * SC, loc="lower right", handletextpad=0.2,
                  borderpad=0.22, labelspacing=0.18)
        ax.tick_params(labelsize=5.6 * SC, width=0.4 * SC, length=1.6 * SC)
    fig.tight_layout(pad=0.3)
    save(fig, out, "fig_scaling")


def fig_ratio(rows, out):
    fig, ax = plt.subplots(figsize=(3.3 * SC, 2.0 * SC))
    base = {}
    for m in ("AOBV-B", "CKP07", "CD01"):
        xs, ys = series(rows, m, "comm")
        base[m] = dict(zip(xs, ys))
    xs, ys = series(rows, "BDFC", "comm")
    ours = dict(zip(xs, ys))
    for m in ("CD01", "CKP07", "AOBV-B"):
        mk, ls, col = STYLE[m]
        common = sorted(set(base[m]) & set(ours))
        ax.plot(common, [base[m][s] / ours[s] for s in common], marker=mk, linestyle=ls,
                color=col, linewidth=0.9 * SC, markersize=3.2 * SC,
                label="versus " + PLAIN[m])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("system size $m$", fontsize=6.2 * SC)
    ax.set_ylabel("communication reduction factor", fontsize=6.2 * SC)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.3 * SC)
    ax.legend(fontsize=5.4 * SC, loc="upper left", framealpha=0.92)
    ax.tick_params(labelsize=5.6 * SC, width=0.4 * SC, length=1.6 * SC)
    fig.tight_layout(pad=0.35)
    save(fig, out, "fig_ratio")


def fig_block(rows, cost, out):
    fig, axes = plt.subplots(1, 2, figsize=(3.42 * SC, 1.35 * SC))
    ax = axes[0]
    for q, col in ((16, "#d62728"), (256, "#1f77b4")):
        for s, mk in ((32, "o"), (64, "s")):
            sub = sorted([(int(r["b"]), float(r["empirical"]), float(r["predicted"]))
                          for r in rows if int(r["q"]) == q and int(r["s"]) == s])
            if not sub:
                continue
            ax.plot([x[0] for x in sub], [x[1] for x in sub], marker=mk, linestyle="none",
                    color=col, markersize=3.6 * SC, label="$q=%d$, $m=%d$" % (q, s))
            ax.plot([x[0] for x in sub], [x[2] for x in sub], linestyle="--",
                    color=col, linewidth=0.7 * SC)
    ax.set_xlabel("block size $b$", fontsize=5.6 * SC)
    ax.set_ylabel("success probability", fontsize=5.6 * SC)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, linewidth=0.3 * SC)
    ax.legend(fontsize=4.2 * SC, loc="lower right", handletextpad=0.3, borderpad=0.25, labelspacing=0.2)
    ax.tick_params(labelsize=5.6 * SC, width=0.4 * SC, length=1.6 * SC)

    ax = axes[1]
    for s, col, mk in ((32, "#2ca02c", "o"), (48, "#1f77b4", "s"), (64, "#d62728", "^")):
        sub = sorted([(int(r["b"]), float(r["comm"]), float(r["rounds"]))
                      for r in cost if int(r["s"]) == s])
        if not sub:
            continue
        ax.plot([x[0] for x in sub], [x[1] for x in sub], marker=mk, color=col,
                linewidth=0.9 * SC, markersize=3.2 * SC, label="$m=%d$" % s)
    ax.set_xlabel("block size $b$", fontsize=5.6 * SC)
    ax.set_ylabel("comm. (elements/party)", fontsize=5.6 * SC)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, linewidth=0.3 * SC)
    ax.legend(fontsize=4.6 * SC, handletextpad=0.3, borderpad=0.25, labelspacing=0.2)
    ax.tick_params(labelsize=5.6 * SC, width=0.4 * SC, length=1.6 * SC)
    fig.tight_layout(pad=0.35)
    save(fig, out, "fig_block")


def fig_latency(rows, out):
    def fit(m, key):
        xs, ys = series(rows, m, key)
        return np.polyfit(np.log(xs), np.log(ys), 1)

    def model(par, ms, rtt, bw):
        lt, lc, lr = par
        return (np.exp(lt[1]) * ms ** lt[0] + np.exp(lr[1]) * ms ** lr[0] * rtt
                + np.exp(lc[1]) * ms ** lc[0] * 8.0 / bw)

    pb = tuple(fit("BDFC", k) for k in ("time", "comm", "rounds"))
    pa = tuple(fit("AOBV-B", k) for k in ("time", "comm", "rounds"))
    ms = np.linspace(32, 128, 25)
    rtts = np.logspace(-4, -1, 25)
    M, R = np.meshgrid(ms, rtts)
    bw = 1e9
    Zb = model(pb, M, R, bw)
    Za = model(pa, M, R, bw)
    fig = plt.figure(figsize=(3.42 * SC, 1.85 * SC))
    gs = fig.add_gridspec(1, 1, left=0.14, right=1.02, bottom=-0.02, top=1.04)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax.plot_surface(M, np.log10(R * 1000.0), np.log10(Za), color="#1f77b4",
                    edgecolor="#0b3d61", linewidth=0.12 * SC, rstride=1, cstride=1)
    ax.plot_surface(M, np.log10(R * 1000.0), np.log10(Zb), color="#d62728",
                    edgecolor="#7a1113", linewidth=0.12 * SC, rstride=1, cstride=1)
    ax.set_xlabel("$m$", fontsize=6.0 * SC, labelpad=-3 * SC)
    ax.set_ylabel("$\\log_{10}$ RTT (ms)", fontsize=6.0 * SC, labelpad=-2 * SC)
    ax.tick_params(labelsize=5.0 * SC, pad=-2.0 * SC)
    ax.view_init(elev=20, azim=-125)
    ax.set_box_aspect((1.3, 1.0, 0.72))
    fig.text(0.055, 0.55, "$\\log_{10}$ latency (s)", rotation=90,
             va="center", ha="left", fontsize=6.0 * SC)
    save(fig, out, "fig_latency")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "axes.linewidth": 0.4 * SC})
    rows = load(os.path.join(args.results, "solvers.csv"))
    bpath = os.path.join(args.results, "blocksize.csv")
    blk = load(bpath) if os.path.exists(bpath) else None
    fig_scaling(rows, args.out, blk)
    fig_ratio(rows, args.out)
    fig_latency(rows, args.out)
    if blk:
        fig_block(blk, load(bpath.replace(".csv", "_cost.csv")), args.out)


if __name__ == "__main__":
    main()
