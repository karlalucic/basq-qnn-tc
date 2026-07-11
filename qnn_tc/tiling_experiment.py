"""Chip tiling: pack K independent copies of the 4-qubit QNN onto one wide
circuit spanning disjoint 4-qubit chains of a Heron heavy-hex lattice.

Idea
----
One test point currently binds one parameter set to one 4-qubit circuit, so
200 test points cost 200 parameter rows on a single 4-qubit PUB. A 156-qubit
Heron chip has room for many disjoint 4-qubit chains. If we place K independent
copies of the ansatz on K disjoint chains (each with its OWN input + weight
parameters), one shot of the wide circuit evaluates K test points at once.
That turns 200 rows into ceil(200/K) rows -> ~1/K QPU time.

Because the tiles live on disjoint qubits with no entangling gates between them,
the joint state is a product state across tiles, so a per-tile mean-Z observable
returns exactly the value that tile's copy would produce on its own. This file
proves that numerically and verifies the wide circuit transpiles SWAP-free.

Run:  .venv/bin/python tiling_experiment.py
"""

from collections import defaultdict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_ibm_runtime.fake_provider import FakeFez

from qnn import build_circuit

TILE_SIZE = 4      # qubits per copy of the ansatz
N_LAYERS = 3       # build_circuit(4, 3, 4)
N_FEATURES = 4


# --------------------------------------------------------------------------- #
# 1. Coupling graph + greedy disjoint-tile packing
# --------------------------------------------------------------------------- #
def adjacency(backend):
    """Undirected adjacency dict {qubit: set(neighbours)} from a backend."""
    adj = defaultdict(set)
    for a, b in {tuple(sorted(e)) for e in backend.coupling_map.get_edges()}:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _find_path(adj, start, avail, size):
    """DFS with backtracking for a simple path of exactly `size` qubits, using
    only qubits in `avail`. Extends toward the neighbour with the fewest onward
    options so hard-to-place corner qubits are consumed first (better packing).
    """
    best = [None]

    def dfs(path, used):
        if best[0] is not None:
            return
        if len(path) == size:
            best[0] = list(path)
            return
        nxts = sorted(
            (x for x in adj[path[-1]] if x in avail and x not in used),
            key=lambda x: sum(1 for y in adj[x] if y in avail and y not in used),
        )
        for x in nxts:
            used.add(x)
            path.append(x)
            dfs(path, used)
            path.pop()
            used.discard(x)

    dfs([start], {start})
    return best[0]


def find_disjoint_tiles(adj, n_qubits, size=TILE_SIZE, buffer=True):
    """Greedily carve out disjoint length-`size` chains.

    Each returned tile is an ordered list [q0, q1, q2, q3] where consecutive
    qubits are physically connected, so the ansatz's linear CZ chain maps onto
    hardware edges with zero SWAPs. When `buffer=True` we also retire every
    neighbour of a chosen tile, guaranteeing at least one unused buffer qubit
    between tiles (crosstalk hygiene).
    """
    avail = set(range(n_qubits))
    tiles = []
    while True:
        # start from the lowest-degree available qubit (uses corners first)
        starts = sorted(avail, key=lambda q: sum(1 for y in adj[q] if y in avail))
        path = None
        for s in starts:
            path = _find_path(adj, s, avail, size)
            if path:
                break
        if not path:
            break
        tiles.append(path)
        for q in path:
            avail.discard(q)
            if buffer:
                for y in adj[q]:
                    avail.discard(y)
    return tiles


def validate_tiles(tiles, adj):
    """Assert tiles are disjoint, valid physical paths, and (with buffer)
    mutually non-adjacent. Returns the number of inter-tile adjacencies."""
    flat = [q for t in tiles for q in t]
    assert len(flat) == len(set(flat)), "tiles overlap"
    for t in tiles:
        for a, b in zip(t, t[1:]):
            assert b in adj[a], f"tile edge {a}-{b} is not a hardware edge"
    inter = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            for a in tiles[i]:
                inter += sum(1 for b in tiles[j] if b in adj[a])
    return inter


# --------------------------------------------------------------------------- #
# 2. Wide circuit: K independent copies of build_circuit(4, 3, 4)
# --------------------------------------------------------------------------- #
def build_wide_circuit(k_tiles):
    """Compose k_tiles copies of the ansatz onto a compact (k_tiles*4)-qubit
    circuit. Copy k occupies logical qubits [4k, 4k+3] and owns a private set of
    input + weight parameters (renamed with a `t{k}_` prefix). Returns
    (wide_qc, tile_inputs, tile_weights) where tile_inputs[k] / tile_weights[k]
    are the parameter objects for copy k, in ansatz order.
    """
    wide = QuantumCircuit(k_tiles * TILE_SIZE)
    tile_inputs, tile_weights = [], []
    for k in range(k_tiles):
        qc, inp, wts, _ = build_circuit(TILE_SIZE, N_LAYERS, N_FEATURES)
        rename = {p: Parameter(f"t{k}_{p.name}") for p in list(inp) + list(wts)}
        qc = qc.assign_parameters(rename, inplace=False)
        wide.compose(qc, qubits=list(range(4 * k, 4 * k + 4)), inplace=True)
        tile_inputs.append([rename[p] for p in inp])
        tile_weights.append([rename[p] for p in wts])
    return wide, tile_inputs, tile_weights


def tile_observables(k_tiles):
    """One mean-Z observable per tile on the compact logical circuit: tile k is
    (1/4) * sum_q Z on logical qubits [4k, 4k+3]. Passed as a list to a single
    PUB so EstimatorV2 returns k_tiles expectation values per parameter row."""
    return [
        SparsePauliOp.from_sparse_list(
            [("Z", [4 * k + q], 1.0 / TILE_SIZE) for q in range(TILE_SIZE)],
            num_qubits=k_tiles * TILE_SIZE,
        )
        for k in range(k_tiles)
    ]


# --------------------------------------------------------------------------- #
# 3. Transpile + verify SWAP-free embedding
# --------------------------------------------------------------------------- #
def transpile_and_verify(wide, tiles, backend):
    """Transpile the wide circuit onto `backend` pinning each logical qubit to
    its tile's physical qubit, then assert 0 SWAPs and cz == K*6. Returns
    (isa_circuit, isa_observable_layout) where the layout maps compact
    observables onto physical qubits via `.apply_layout`."""
    k = len(tiles)
    phys = [q for t in tiles for q in t]
    layout = {wide.qubits[i]: phys[i] for i in range(len(phys))}
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, initial_layout=layout
    )
    isa = pm.run(wide)
    ops = isa.count_ops()
    n_swap = ops.get("swap", 0)
    n_cz = ops.get("cz", 0)
    print(f"  wide ISA: depth {isa.depth()}, cz {n_cz} (expect {6 * k}), "
          f"swap {n_swap}")
    assert n_swap == 0, f"expected 0 SWAPs, got {n_swap}"
    assert n_cz == 6 * k, f"expected {6 * k} CZ, got {n_cz}"
    return isa


# --------------------------------------------------------------------------- #
# 4. Calibration-aware tile ranking (mitigation for bad-qubit tiles)
# --------------------------------------------------------------------------- #
def rank_tiles_by_calibration(tiles, backend):
    """Score each tile by summed readout error + summed CZ error along its
    chain (lower = better). Lets us keep only the best tiles when we need fewer
    than the max, dodging poorly calibrated regions of the chip."""
    target = backend.target
    ro = {q: target["measure"][(q,)].error for q in range(backend.num_qubits)}

    def cz_err(a, b):
        for key in ((a, b), (b, a)):
            inst = target["cz"].get(key)
            if inst is not None:
                return inst.error
        return 0.0

    scored = []
    for t in tiles:
        s = sum(ro[q] for q in t) + sum(cz_err(a, b) for a, b in zip(t, t[1:]))
        scored.append((s, t))
    scored.sort(key=lambda x: x[0])
    return scored


# --------------------------------------------------------------------------- #
# 5. Correctness: tiled evs must equal separate single-circuit runs
# --------------------------------------------------------------------------- #
def correctness_check(k=3, seed=0, tol=1e-6):
    """Bind k different test points to a k-tile wide circuit and confirm its k
    per-tile expectation values match k independent single-circuit runs.
    Uses the compact logical circuits on an exact Aer statevector estimator
    (precision=0), so agreement should be at machine precision."""
    rng = np.random.default_rng(seed)
    _, inp0, wts0, _ = build_circuit(TILE_SIZE, N_LAYERS, N_FEATURES)
    weights = rng.uniform(-0.3, 0.3, len(wts0))          # shared trained weights
    points = rng.uniform(-1.0, 1.0, (k, N_FEATURES))     # k distinct test points

    est = AerEstimator()

    # --- wide circuit, one PUB, k observables ---
    wide, tin, twt = build_wide_circuit(k)
    bind = {}
    for kk in range(k):
        bind.update({p: v for p, v in zip(tin[kk], points[kk])})
        bind.update({p: v for p, v in zip(twt[kk], weights)})
    vals = np.array([bind[p] for p in wide.parameters])
    tiled = np.asarray(
        est.run([(wide, tile_observables(k), vals)], precision=0.0)
        .result()[0].data.evs
    ).ravel()

    # --- k independent single-circuit runs ---
    single = []
    for kk in range(k):
        qc, inp, wts, obs = build_circuit(TILE_SIZE, N_LAYERS, N_FEATURES)
        b = {p: v for p, v in zip(list(inp) + list(wts),
                                  list(points[kk]) + list(weights))}
        vv = np.array([b[p] for p in qc.parameters])
        single.append(float(np.asarray(
            est.run([(qc, obs, vv)], precision=0.0).result()[0].data.evs
        ).ravel()[0]))
    single = np.array(single)

    max_diff = float(np.max(np.abs(tiled - single)))
    ok = max_diff < tol
    print(f"  tiled  evs: {np.array2string(tiled,  precision=6)}")
    print(f"  single evs: {np.array2string(single, precision=6)}")
    print(f"  max abs diff: {max_diff:.2e}  (tol {tol:.0e}) -> "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok, "tiled and single-circuit expectation values disagree"
    return max_diff


# --------------------------------------------------------------------------- #
# 6. Throughput estimate
# --------------------------------------------------------------------------- #
def throughput_estimate(k, n_points=200, base_points=50, base_seconds=41.0):
    """QPU-time scaling: K tiles need ceil(n_points/K) rows instead of n_points.
    Baseline: base_points points measured in base_seconds on ibm_fez."""
    per_point = base_seconds / base_points
    rows_single = n_points
    rows_tiled = int(np.ceil(n_points / k))
    t_single = per_point * rows_single
    t_tiled = per_point * rows_tiled
    print(f"  baseline: {base_points} pts -> {base_seconds:.0f}s "
          f"({per_point:.2f} s/row)")
    print(f"  {n_points} pts single-copy : {rows_single} rows -> {t_single:.0f}s")
    print(f"  {n_points} pts, K={k} tiles : {rows_tiled} rows -> {t_tiled:.1f}s "
          f"(~{rows_single / rows_tiled:.0f}x faster)")
    return t_single, t_tiled


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    backend = FakeFez()
    adj = adjacency(backend)
    print(f"backend {backend.name}: {backend.num_qubits} qubits, "
          f"{len({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})} "
          f"edges (heavy-hex)\n")

    print("[1] greedy disjoint-tile packing")
    tiles = find_disjoint_tiles(adj, backend.num_qubits, buffer=True)
    tiles_nb = find_disjoint_tiles(adj, backend.num_qubits, buffer=False)
    inter = validate_tiles(tiles, adj)
    K = len(tiles)
    print(f"  buffered  : K={K} tiles, {4 * K} qubits, "
          f"inter-tile adjacencies={inter}")
    print(f"  packed    : K={len(tiles_nb)} tiles (no buffer, upper bound)")
    print(f"  max K (crosstalk-safe) = {K}\n")

    print("[2] wide circuit build + SWAP-free transpile")
    wide, _, _ = build_wide_circuit(K)
    print(f"  wide logical: {wide.num_qubits} qubits, "
          f"{wide.num_parameters} parameters")
    transpile_and_verify(wide, tiles, backend)
    print()

    print("[3] calibration ranking (mitigation for bad tiles)")
    ranked = rank_tiles_by_calibration(tiles, backend)
    print(f"  best tile  score {ranked[0][0]:.4f} on {ranked[0][1]}")
    print(f"  worst tile score {ranked[-1][0]:.4f} on {ranked[-1][1]}")
    print(f"  -> when <K points remain, keep the best-scoring tiles\n")

    print("[4] correctness check (K=3, exact Aer statevector)")
    correctness_check(k=3)
    print()

    print("[5] throughput estimate")
    throughput_estimate(K)


if __name__ == "__main__":
    main()
