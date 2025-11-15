#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Differential Evolution for EV Charging/Discharging Scheduling
- 20 individuals, 96 time slots (15-min intervals)
- Decision variable: -1 (discharge), 0 (idle), 1 (charge)
- Objective: minimise cost = charge_cost - discharge_revenue
"""

import numpy as np
import time

# ===================================================================
# 1. DE PARAMETERS
# ===================================================================
NP        = 20          # population size
D         = 96          # number of time slots (24 h × 4)
F         = 0.30        # mutation factor
CR        = 0.60        # crossover probability
MAX_ITER  = 10          # max generations
STRATEGY  = 5           # DE/rand/1 with per-generation-dither

MIN_VAL = -1
MAX_VAL =  1

# EV PARAMETERS
BC       = 50 / 4          # battery capacity per slot (kWh)
CHG_EFF  = 90 / 100        # charging efficiency
CCR      = 4 / 4           # charging power (kW)
DCR      = 1 / 4           # discharging power (kW)
CCC      = 7               # cost of charging ($/kWh)
DCC      = 3               # revenue of discharging ($/kWh)

# Time vector (0 … 23.45 h)
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
# 2. EV DATA (replace with your real matrices)
# ===================================================================
def generate_dummy_ev_data(num_evs=5):
    """Creates a (num_evs, 4*NP) matrix with random arrival/departure/SOC."""
    np.random.seed(42)
    data = np.zeros((num_evs, 4 * NP))
    for i in range(NP):
        base = i * 4
        data[:, base]     = np.random.uniform(6, 10, num_evs)   # arrival 6-10 am
        data[:, base + 1] = np.random.uniform(16, 20, num_evs) # departure 4-8 pm
        data[:, base + 2] = np.random.uniform(0.2, 0.6, num_evs)# SOC arrival
        data[:, base + 3] = np.random.uniform(0.8, 1.0, num_evs)# SOC departure
    return data

ev_data      = generate_dummy_ev_data()      # training / initialization
ev_data_test = generate_dummy_ev_data()      # evaluation inside DE loop

# ===================================================================
# 3. COST FUNCTION
# ===================================================================
def calculate_ev_cost(pop, ev_matrix):
    """
    pop        : (NP, D)  → schedule (-1,0,1)
    ev_matrix  : (num_evs, 4*NP)
    returns    : np.array of shape (NP,) with costs
    """
    costs = np.zeros(NP)
    num_evs = ev_matrix.shape[0]

    for i in range(NP):
        base = i * 4
        arr_times = ev_matrix[:, base]
        dep_times = ev_matrix[:, base + 1]
        soc_arr   = ev_matrix[:, base + 2]
        soc_dep   = ev_matrix[:, base + 3]

        arr_max = np.max(arr_times)
        dep_min = np.min(dep_times)
        soc_arr_max = np.max(soc_arr)
        soc_dep_min = np.min(soc_dep)

        # required energy (kWh)
        soc_diff = soc_dep_min - soc_arr_max
        if soc_diff <= 0:
            costs[i] = 0.0
            continue
        eng_req = (soc_diff * BC) / CHG_EFF

        # clamp times
        arr_max = min(arr_max, 23.45)
        dep_min = min(dep_min, 23.45)

        cost = 0.0
        remaining = eng_req

        for t in range(D):
            t_val = TIME_SLOTS[t]
            action = pop[i, t]

            if t_val >= arr_max and t_val < dep_min + 0.15:
                if action == 1 and remaining > 0:          # charge
                    charge = CCR
                    cost += CCC * charge
                    remaining -= charge
                    if remaining < 0:
                        remaining = 0
                elif action == -1 and remaining > 0:        # discharge
                    discharge = DCR
                    cost -= (DCC + discharge)               # revenue
                    remaining += discharge
                # action == 0 → idle
            else:
                pop[i, t] = 0                               # force idle outside window

        # huge penalty if energy demand is not satisfied
        costs[i] = cost if remaining <= 0 else 1e6

    return costs

# ===================================================================
# 4. INITIAL POPULATION
# ===================================================================
pop = np.random.choice([-1, 0, 1], size=(NP, D))
current_costs = calculate_ev_cost(pop, ev_data)
best_idx = np.argmin(current_costs)
best_mem = pop[best_idx].copy()
best_val = current_costs[best_idx]

# ===================================================================
# 5. MAIN DE LOOP
# ===================================================================
start = time.time()

print(f"{'Iter':>6}  {'Best Cost':>15}  {'First 10 actions':>40}")
print("-" * 80)

for gen in range(1, MAX_ITER + 1):
    old_pop = pop.copy()

    for i in range(NP):
        # ---- mutation -------------------------------------------------
        idxs = [j for j in range(NP) if j != i]
        a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

        if STRATEGY == 5:                         # per-generation dither
            f_mut = F * (1 - 0.999 * np.random.rand()) + F
            mutant = c + f_mut * (a - b)
        else:
            mutant = c + F * (a - b)

        # ---- crossover ------------------------------------------------
        cross = np.random.rand(D) < CR
        if not np.any(cross):
            cross[np.random.randint(D)] = True
        trial = np.where(cross, mutant, old_pop[i])

        # ---- boundary handling (bounce-back) -------------------------
        trial = np.clip(trial, MIN_VAL, MAX_VAL)
        high = trial > MAX_VAL
        low  = trial < MIN_VAL
        origin = c if STRATEGY == 5 else a
        trial[high] = MAX_VAL + np.random.rand(high.sum()) * (origin[high] - MAX_VAL)
        trial[low]  = MIN_VAL + np.random.rand(low.sum())  * (origin[low]  - MIN_VAL)
        trial = np.clip(trial, MIN_VAL, MAX_VAL)

        # ---- evaluation -----------------------------------------------
        trial_cost = calculate_ev_cost(trial.reshape(1, -1), ev_data_test)[0]

        # ---- selection ------------------------------------------------
        if trial_cost <= current_costs[i]:
            pop[i] = trial
            current_costs[i] = trial_cost
            if trial_cost < best_val:
                best_val = trial_cost
                best_mem = trial.copy()

    print(f"{gen:6d}  {best_val:15.6f}  {best_mem[:10]}")

print("-" * 80)
print(f"Finished in {time.time() - start:.2f} s")
print(f"Best cost : {best_val:.6f}")
print(f"Best schedule (first 20 slots): {best_mem[:20]}")