#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Differential Evolution for EV Charging/Discharging Scheduling (Version 2)
- Population: 20
- Dimensions: 96 time slots
- F=0.4, CR=0.7
- Strategy: 5 (DE/rand/1 with per-generation-dither)
- Max Iter: 10
"""

import numpy as np
import time

# ===================================================================
# 1. DE Parameters
# ===================================================================
NP = 20           # Population size
D = 96            # Number of time slots
F = 0.40          # Mutation factor
CR = 0.70         # Crossover probability
MAX_ITER = 10     # Max generations
STRATEGY = 5      # DE/rand/1 with per-generation-dither

# Bounds: -1, 0, 1
MIN_VAL = -1
MAX_VAL = 1

# EV Parameters
BC = 50 / 4       # Battery capacity per slot (kWh)
CHG_EFF = 90 / 100  # 90%
CCR = 4 / 4       # Charging rate (kW)
DCR = 1 / 4       # Discharging rate (kW)
CCC = 7           # Charging cost ($/kWh)
DCC = 3           # Discharge revenue ($/kWh)

# Time vector (24 hours in 15-min steps)
TIME_SLOTS = np.array([
    0, 0.15, 0.30, 0.45, 1.00, 1.15, 1.30, 1.45, 2.00, 2.15, 2.30, 2.45,
    3.00, 3.15, 3.30, 3.45, 4.00, 4.15, 4.30, 4.45, 5.00, 5.15, 5.30, 5.45,
    6.00, 6.15, 6.30, 6.45, 7.00, 7.15, 7.30, 7.45, 8.00, 8.15, 8.30, 8.45,
    9.00, 9.15, 9.30, 9.45, 10.00, 10.15, 10.30, 10.45, 11.00, 11.15, 11.30, 11.45,
    12.00, 12.15, 12.30, 12.45, 13.00, 13.15, 13.30, 13.45, 14.00, 14.15, 14.30, 14.45,
    15.00, 15.15, 15.30, 15.45, 16.00, 16.15, 16.30, 16.45, 17.00, 17.15, 17.30, 17.45,
    18.00, 18.15, 18.30, 18.45, 19.00, 19.15, 19.30, 19.45, 20.00, 20.15, 20.30, 20.45,
    21.00, 21.15, 21.30, 21.45, 22.00, 22.15, 22.30, 22.45, 23.00, 23.15, 23.30, 23.45
])

# ===================================================================
# 2. Load EV Data (Replace with real data)
# ===================================================================
# Example: 5 EVs, columns: [arrival, departure, SOC_arr, SOC_dep]
# Shape: (num_EVs, 4*NP) → for 20 individuals
# Here: generate dummy data for demo

def generate_dummy_ev_data(num_evs=5):
    np.random.seed(42)
    data = np.zeros((num_evs, 4 * NP))
    for i in range(NP):
        base = i * 4
        # Random arrival: 6–10 AM
        arr = np.random.uniform(6, 10, num_evs)
        # Random departure: 16–20 PM
        dep = np.random.uniform(16, 20, num_evs)
        # SOC: 20–60% arrival, 80–100% departure
        soc_arr = np.random.uniform(0.2, 0.6, num_evs)
        soc_dep = np.random.uniform(0.8, 1.0, num_evs)
        data[:, base] = arr
        data[:, base + 1] = dep
        data[:, base + 2] = soc_arr
        data[:, base + 3] = soc_dep
    return data

ev_data = generate_dummy_ev_data()        # mat67
ev_data_test = generate_dummy_ev_data()   # mat2916

# ===================================================================
# 3. EV Energy Requirement & Cost Function
# ===================================================================
def calculate_ev_cost(pop, ev_matrix):
    """
    pop: (NP, D) → -1, 0, 1 schedule
    ev_matrix: (num_evs, 4*NP)
    Returns: cost vector (NP,)
    """
    costs = np.zeros(NP)
    num_evs = ev_matrix.shape[0]

    for i in range(NP):
        # Extract EV data for individual i
        base = i * 4
        arr_times = ev_matrix[:, base]
        dep_times = ev_matrix[:, base + 1]
        soc_arr = ev_matrix[:, base + 2]
        soc_dep = ev_matrix[:, base + 3]

        arr_max = np.max(arr_times)
        dep_min = np.min(dep_times)
        soc_arr_max = np.max(soc_arr)
        soc_dep_min = np.min(soc_dep)

        # Energy required (kWh)
        soc_diff = soc_dep_min - soc_arr_max
        if soc_diff <= 0:
            costs[i] = 0
            continue
        eng_req = (soc_diff * BC) / CHG_EFF

        # Clamp times
        arr_max = min(arr_max, 23.45)
        dep_min = min(dep_min, 23.45)

        cost = 0.0
        remaining_energy = eng_req

        for t in range(D):
            time_val = TIME_SLOTS[t]
            action = pop[i, t]

            if time_val >= arr_max and time_val < dep_min + 0.15:
                if action == 1 and remaining_energy > 0:
                    charge = CCR
                    cost += CCC * charge
                    remaining_energy -= charge
                    if remaining_energy < 0:
                        remaining_energy = 0
                elif action == -1 and remaining_energy > 0:
                    discharge = DCR
                    cost -= DCC * discharge  # revenue
                    remaining_energy += discharge
                # action == 0 → no change
            else:
                pop[i, t] = 0  # force idle outside window

        if remaining_energy > 0:
            costs[i] = 1e6  # penalty
        else:
            costs[i] = cost

    return costs

# ===================================================================
# 4. DE Initialization
# ===================================================================
pop = np.random.choice([-1, 0, 1], size=(NP, D))
current_costs = calculate_ev_cost(pop, ev_data)
best_idx = np.argmin(current_costs)
best_mem = pop[best_idx].copy()
best_val = current_costs[best_idx]

# ===================================================================
# 5. Main DE Loop
# ===================================================================
start_time = time.time()

print(f"{'Iter':>6}  {'Best Cost':>15}  {'Best Member (first 10)':>40}")
print("-" * 80)

for gen in range(1, MAX_ITER + 1):
    old_pop = pop.copy()

    for i in range(NP):
        # Mutation
        idxs = [idx for idx in range(NP) if idx != i]
        a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

        if STRATEGY == 5:  # DE/rand/1 with per-generation-dither
            f_mut = (1 - F) * np.random.rand() + F
            mutant = c + f_mut * (a - b)
        else:
            mutant = c + F * (a - b)

        # Crossover
        cross_points = np.random.rand(D) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, D)] = True
        trial = np.where(cross_points, mutant, old_pop[i])

        # Boundary handling (bounce-back)
        trial = np.clip(trial, MIN_VAL, MAX_VAL)
        exceed_high = trial > MAX_VAL
        exceed_low = trial < MIN_VAL
        origin = c if STRATEGY == 5 else a
        trial[exceed_high] = MAX_VAL + np.random.rand(np.sum(exceed_high)) * (origin[exceed_high] - MAX_VAL)
        trial[exceed_low] = MIN_VAL + np.random.rand(np.sum(exceed_low)) * (origin[exceed_low] - MIN_VAL)
        trial = np.clip(trial, MIN_VAL, MAX_VAL)

        # Evaluate trial
        trial_cost = calculate_ev_cost(trial.reshape(1, -1), ev_data_test)[0]

        # Selection
        if trial_cost <= current_costs[i]:
            pop[i] = trial
            current_costs[i] = trial_cost
            if trial_cost < best_val:
                best_val = trial_cost
                best_mem = trial.copy()

    print(f"{gen:6d}  {best_val:15.6f}  {best_mem[:10]}")

print("-" * 80)
print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
print(f"Best Cost: {best_val:.6f}")
print(f"Best Schedule (first 20 slots): {best_mem[:20]}")