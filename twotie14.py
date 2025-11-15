#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Power-flow + network reconfiguration for IEEE 33-bus system.
Returns: active loss (kW), reactive loss (kVar)
Used as objective in Differential Evolution.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List

# ----------------------------------------------------------------------
# 1. System Data (from your MATLAB script)
# ----------------------------------------------------------------------
# Load profile (96 time steps)
ld33 = np.array([
    1, 0.796852378954214, 0.905685523350686, 0.772852355723896, 0.696457104419982,
    0.969665398716195, 0.834907826074480, 0.619261495184992, 0.963817904941169,
    0.762981610360049, 0.597804198647279, 0.612916593412120, 0.817775861764106,
    0.8618744294621711, 0.885600853352678, 0.802574797297182, 0.737132906539590,
    0.834320806053298, 0.949627799829399, 0.603900887507537, 0.582808963497374,
    0.919960825288942, 0.791553749048017, 0.739284683846444, 0.971237733701583,
    0.792226426143864, 0.849970189866794, 0.882752239143718, 0.925919842280361,
    0.809659018808607, 0.753455078934539, 0.818126008584402, 0.915317475175245,
    0.796112806535499, 0.597141352971931, 0.845548747153255, 0.747801591753298,
    0.645493943885950, 0.632504208703770, 0.667229921967480, 0.870105180333912,
    0.688613495854660, 0.859043879989780, 0.895840808074453, 0.624303744672319,
    0.776853205678323, 0.921556193759485, 0.991667492629801, 0.584356418635378,
    0.725591157420363, 0.711469703844786, 0.726220164017709, 0.892192077463970,
    0.872556582457655, 0.805508459090065, 0.634881701422424, 0.636470658086701,
    0.866083467332462, 0.882329049174814, 0.731330294678665, 0.939655958378722,
    0.878153388428876, 0.972901651915848, 0.943200437249371, 0.741321178322164,
    0.652102508365957, 0.752220876607755, 0.611631427888053, 0.874173081109239,
    0.767604124538325, 0.961503625190938, 0.867785290657663, 0.603448116686367,
    0.809650110784813, 0.684755283465572, 0.915571136574015, 0.930240332501103,
    0.991035147547937, 0.896449179129082, 0.802681233548710, 0.653787872553817,
    0.613136089038111, 0.624067070663698, 0.864789728506582, 0.889733090283428,
    0.759474491440890, 0.611193350919206, 0.944314251732924, 0.688654119784208,
    0.850090868020730, 0.735558263608668, 0.676477491079994, 0.892908085479423,
    0.890380568691105, 0.947387217125089, 0.888747678652145
])

# Node data: [bus, P(kW), Q(kVar), B(pu)]
M = np.array([
    [1,   0,   0, 0],
    [2, 100,  60, 0],
    [3,  90,  40, 0],
    [4, 120,  80, 0],
    [5,  60,  30, 0],
    [6,  60,  20, 0],
    [7, 200, 100, 0],
    [8, 200, 100, 0],
    [9,  60,  20, 0],
    [10, 60,  20, 0],
    [11, 45,  30, 0],
    [12, 60,  35, 0],
    [13, 60,  35, 0],
    [14,120,  80, 0],
    [15, 60,  10, 0],
    [16, 60,  20, 0],
    [17, 60,  20, 0],
    [18, 90,  40, 0],
    [19, 90,  40, 0],
    [20, 90,  40, 0],
    [21, 90,  40, 0],
    [22, 90,  40, 0],
    [23, 90,  50, 0],
    [24,420, 200, 0],
    [25,420, 200, 0],
    [26, 60,  25, 0],
    [27, 60,  25, 0],
    [28, 60,  20, 0],
    [29,120,  70, 0],
    [30,200, 600, 0],
    [31,150,  70, 0],
    [32,210, 100, 0],
    [33, 60,  40, 0]
], dtype=float)

# Original line data: [line_no, from, to, R(ohm), X(ohm)]
l_original = np.array([
    [1,  1,  2, 0.0922, 0.0470],
    [2,  2,  3, 0.4930, 0.2511],
    [3,  3,  4, 0.3660, 0.1864],
    [4,  4,  5, 0.3811, 0.1941],
    [5,  5,  6, 0.8190, 0.7070],
    [6,  6,  7, 0.1872, 0.6188],
    [7,  7,  8, 1.7114, 1.2351],
    [8,  8,  9, 1.0300, 0.7400],
    [9,  9, 10, 1.0400, 0.7400],
    [10,10, 11, 0.1966, 0.0650],
    [11,11, 12, 0.3744, 0.1238],
    [12,12, 13, 1.4680, 1.1550],
    [13,13, 14, 0.5416, 0.7129],
    [14,14, 15, 0.5910, 0.5260],
    [15,15, 16, 0.7463, 0.5450],
    [16,16, 17, 1.2890, 1.7210],
    [17,17, 18, 0.7320, 0.5740],
    [18, 2, 19, 0.1640, 0.1565],
    [19,19, 20, 1.5042, 1.3554],
    [20,20, 21, 0.4095, 0.4784],
    [21,21, 22, 0.7089, 0.9373],
    [22, 3, 23, 0.4512, 0.3083],
    [23,23, 24, 0.8980, 0.7091],
    [24,24, 25, 0.8960, 0.7011],
    [25, 6, 26, 0.2030, 0.1034],
    [26,26, 27, 0.2842, 0.1447],
    [27,27, 28, 1.0590, 0.9337],
    [28,28, 29, 0.8042, 0.7006],
    [29,29, 30, 0.5075, 0.2585],
    [30,30, 31, 0.9744, 0.9630],
    [31,31, 32, 0.3105, 0.3619],
    [32,32, 33, 0.3410, 0.5302]
])

# Tie-lines: [tie_no, from, to, R, X]
Tielines = np.array([
    [1,  8, 21, 2.0, 2.0],
    [2,  9, 15, 2.0, 2.0],
    [3, 12, 22, 2.0, 2.0],
    [4, 25, 29, 0.5, 0.5],
    [5, 18, 33, 0.5, 0.5]
])

# System base
MVAb = 100.0
KVb = 12.66
Zb = (KVb ** 2) / MVAb

# ----------------------------------------------------------------------
# 2. Reconfiguration Function
# ----------------------------------------------------------------------
def reconfigure_network(tie_idx: int, open_branch: int) -> Tuple[np.ndarray, bool]:
    """
    Reconfigure network:
        - Close tie-line `tie_idx` (1..5)
        - Open branch `open_branch` (1..32)
    Returns:
        l_new: updated line data (N_branch x 5)
        is_radial: True if network is radial
    """
    if tie_idx < 1 or tie_idx > 5:
        raise ValueError("tie_idx must be 1 to 5")
    if open_branch < 1 or open_branch > 32:
        raise ValueError("open_branch must be 1 to 32")

    l = l_original.copy()
    tie = Tielines[tie_idx - 1]

    # Add tie-line as line 33
    l = np.vstack([l, np.array([33, tie[1], tie[2], tie[3], tie[4]])])

    # Open the branch
    l[open_branch - 1, 1:5] = 0  # set from, to, R, X = 0

    # Build directed graph to check radiality
    G = nx.DiGraph()
    for row in l:
        fr, to = int(row[1]), int(row[2])
        if fr > 0 and to > 0:
            G.add_edge(fr, to)

    # Must be a tree: connected + no cycles + 33 nodes
    if len(G) != 33:
        return l, False
    if not nx.is_weakly_connected(G.to_undirected()):
        return l, False
    if nx.number_of_edges(G) != 32:
        return l, False
    return l, True

# ----------------------------------------------------------------------
# 3. Backward-Forward Sweep Power Flow
# ----------------------------------------------------------------------
def bfs_power_flow(l_active: np.ndarray, time_step: int = 0) -> Tuple[float, float]:
    """
    Backward-Forward Sweep for one time step.
    Returns: P_loss (kW), Q_loss (kVar)
    """
    # Per-unit loads at this time step
    scale = ld33[time_step]
    P_load = M[:, 1] * scale / (1000 * MVAb)  # pu
    Q_load = M[:, 2] * scale / (1000 * MVAb)  # pu

    # Active branches only
    active = l_active[:, 1] > 0
    br = len(l_active[active])
    if br == 0:
        return np.inf, np.inf

    l_br = l_active[active]
    R_pu = l_br[:, 3] / Zb
    X_pu = l_br[:, 4] / Zb
    Z_pu = R_pu + 1j * X_pu

    # Build graph
    G = nx.DiGraph()
    for i, row in enumerate(l_br):
        fr, to = int(row[1]), int(row[2])
        G.add_edge(fr, to, idx=i)

    # Identify root, branches, leaves
    root = 1
    leaves = [n for n in G if G.out_degree(n) == 0]
    branch_nodes = [n for n in G if G.out_degree(n) > 1]

    # Initialize
    V = np.ones(34, dtype=complex)  # index 1..33
    Ibr = np.zeros(br, dtype=complex)

    # Backward sweep: compute branch currents
    for leaf in leaves:
        stack = [leaf]
        while stack:
            node = stack[-1]
            pred = list(G.predecessors(node))
            if not pred:
                stack.pop()
                continue
            pred = pred[0]
            edge_idx = G[pred][node]['idx']
            # Load current at node
            I_load = np.conj(complex(P_load[node-1], Q_load[node-1]) / np.conj(V[node]))
            # Total current entering node
            I_in = I_load
            for child in G.successors(node):
                child_edge = G[node][child]['idx']
                I_in += Ibr[child_edge]
            Ibr[edge_idx] = I_in
            if node != root:
                stack.append(pred)
            else:
                stack.pop()

    # Forward sweep: update voltages
    V[root] = 1.0 + 0j
    for edge_idx, (fr, to) in enumerate(zip(l_br[:, 1], l_br[:, 2])):
        fr, to = int(fr), int(to)
        V[to] = V[fr] - Ibr[edge_idx] * Z_pu[edge_idx]

    # Check voltage limits
    Vmag = np.abs(V[1:])
    if not (np.all(Vmag >= 0.8) and np.all(Vmag <= 1.05)):
        return np.inf, np.inf

    # Compute losses
    Ibr_mag = np.abs(Ibr)
    P_loss_pu = np.sum(Ibr_mag**2 * R_pu)
    Q_loss_pu = np.sum(Ibr_mag**2 * X_pu)

    P_loss_kW = P_loss_pu * 100 * 1000  # pu → MW → kW
    Q_loss_kVar = Q_loss_pu * 100 * 1000

    return P_loss_kW, Q_loss_kVar

# ----------------------------------------------------------------------
# 4. Main Fitness Function (for DE)
# ----------------------------------------------------------------------
def network_loss_fitness(x: np.ndarray, time_step: int = 0) -> float:
    """
    x = [tie_line_idx, open_branch_idx]  (integers)
    Returns total active loss (kW) over the time step.
    If invalid → returns large penalty.
    """
    tie_idx = int(x[0])
    open_branch = int(x[1])

    l_new, is_radial = reconfigure_network(tie_idx, open_branch)
    if not is_radial:
        return 1e6  # penalty

    P_loss, Q_loss = bfs_power_flow(l_new, time_step)
    if np.isinf(P_loss):
        return 1e6

    return P_loss * 0.25  # as in MATLAB: rpl = psum * 0.25

# ----------------------------------------------------------------------
# 5. Example Usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example: tie-line 1, open branch 18
    x_test = np.array([1, 18])
    loss = network_loss_fitness(x_test, time_step=0)
    print(f"Active Loss (kW): {loss:.2f}")

    # Plot network
    l_new, _ = reconfigure_network(1, 18)
    G = nx.Graph()
    for row in l_new:
        fr, to = int(row[1]), int(row[2])
        if fr > 0 and to > 0:
            G.add_edge(fr, to)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
    plt.title("Reconfigured 33-Bus Network")
    plt.show()