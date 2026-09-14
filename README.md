# Leakage-Free Post-Quantum Threshold Signatures from Oil and Vinegar

The package contains a finite field layer for `GF(2^e)` with subfield embeddings, an arithmetic black box (ABB) that emulates `N` parties in one process and counts communication, rounds, preprocessing and secret shared state, six distributed samplers for a secret shared linear system `A x = y`, namely the two masking protocols of Cozzo–Smart and of Celi–Escudero–Niot, the two characteristic polynomial protocols of the Cramer–Damgård and Cramer–Kiltz–Padró line lifted to characteristic two, the Samuelson–Berkowitz protocol, and **BDFC**, the block division free condensation of this work, a threshold Oil and Vinegar signature with dealer-free key generation, a batched public key check, a message independent preprocessing phase and a one-round online phase, robust opening with Berlekamp–Welch decoding, which reconstructs a value and names every deviating signer.

## Running

Python 3.10 or later with `numpy` is required, `matplotlib` is needed only for the figures.
```bash
python3 tests/test_all.py
```
This runs 48 correctness checks as field arithmetic, the Beaver layer, all four
determinant subroutines, all six samplers on square and rectangular systems,
the one sided soundness of the condensation chain on singular matrices, plain
UOV signing and verification, distributed key generation with its batched
check, threshold signing, and robust opening with cheater identification.
A minimal end-to-end example:
```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from tsov.abb import ABB
from tsov.uov import UOV
from tsov.threshold import ThresholdUOV
u = UOV('UOV-Ip', seed=1)
abb = ABB(u.gf, 3, 1)
th = ThresholdUOV(abb, u)
P1, P2 = u.public_seed_material()
sO, P3 = th.dkg(P1, P2)
assert th.verify_key(P1, P2, P3, sO, reps=2)
km = th.key_material(P1, P2, sO)
pre = th.offline(km)
salt = b'0'*24
sig = th.online(pre, b'hello', salt)
print('valid:', u.verify((P1,P2,P3), b'hello', (sig, salt)))
print('online cost:', abb.rounds, 'rounds so far')
"
```

```bash
bash run_all.sh
```

The script runs the four benchmarks in sequence and then builds the results.
It is single threaded and takes roughly twenty minutes on a laptop; the
individual steps are

| command | output | content |
|---|---|---|
| `python3 benchmarks/bench_solvers.py` | `results/solvers.csv` | six samplers over `GF(2^8)` for `m` from 8 to 96 |
| `python3 benchmarks/bench_blocksize.py` | `results/blocksize.csv`, `results/blocksize_cost.csv` | success probability and cost against the block size |
| `python3 benchmarks/bench_threshold.py` | `results/threshold.csv` | key generation, check, preprocessing and online phase |
| `python3 benchmarks/bench_mayo.py` | `results/mayo.csv` | MAYO system shapes over `GF(2^4)` |
| `python3 benchmarks/bench_robust.py` | `results/robust.csv` | robust opening and cheater identification |
| `python3 benchmarks/make_figures.py` | `figures/*.eps`, `figures/*.pdf` | all figures of the paper |


## Files and content

```
tsov/gf.py               GF(2^e) arithmetic, linear algebra, subfield embedding
tsov/abb.py              arithmetic black box, additive sharing, Beaver products
tsov/masked.py           reusable matrix masks, one opening per fresh operand
tsov/ring.py             truncated polynomial ring F_q[y]/(y^D) and its ABB layer
tsov/robust.py           Shamir sharing and Berlekamp-Welch robust opening
tsov/uov.py              plain UOV and MAYO parameter sets, keygen, sign, verify
tsov/threshold.py        DKG, batched key check, preprocessing, online signing
tsov/solvers/common.py   Samuelson-Berkowitz, small division free det and adjugate
tsov/solvers/masking.py  Cozzo-Smart and Celi-Escudero-Niot samplers
tsov/solvers/newton.py   generalized Newton determinant, naive and baby step giant step
tsov/solvers/aobv.py     solve with secure determinant wrappers
tsov/solvers/ours.py     block division free condensation (BDFC)
tests/test_all.py        correctness checks
```

## Measurements

* **Communication** is the number of `GF(2^e)` elements one party sends,
  counted inside the ABB at every opening.
* **Rounds** is the number of synchronous batches of openings.
* **Preprocessing** is the number of correlated random elements consumed, and
  the time spent producing them is excluded from the reported running time, so
  that the comparison is between the online parts of the protocols.
* **Memory** is the peak number of bytes of secret shared state alive at one
  moment, divided by the number of parties. It is tracked by the ABB itself and
  is therefore independent of how the field arithmetic is implemented.

## Parameters

`tsov/uov.py` carries the deployed parameter sets. `UOV-Ip`, `UOV-III` and
`UOV-V` use `q = 256` with `m = o = 44, 72, 96`; `MAYO-1`, `MAYO-3` and `MAYO-5`
use `q = 16` with `m = 64, 96, 128` and a system of `m` equations in `k*o`
unknowns. `TEST` is a small set for fast checks.

The block size `b` of BDFC defaults to 4, which minimizes the product of
communication and rounds over `GF(2^8)`. Over `GF(2^4)` the pivot blocks are
singular too often, so `tsov/solvers/ours.py` also provides `solve_subfield`,
which runs the condensation over the quadratic extension while keeping the
solution in the base field.

