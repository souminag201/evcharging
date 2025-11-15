# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# DE for EV charging (96 time-slots, 20 individuals)
# Decision variables : -1 (discharge), 0 (idle), 1 (charge)
# Uses the exact cost logic from twotie14 / twotie142
# Prints the whole population matrix (96×20) at the end.
# """

# import numpy as np
# import pandas as pd
# import time
# import matplotlib.pyplot as plt

# # ----------------------------------------------------------------------
# # 1. IMPORT THE COST FUNCTIONS (twotie14 → rpl, twotie142 → rpl2)
# # ----------------------------------------------------------------------
# # Save the two blocks you posted as twotie14.py and twotie142.py
# # in the same folder and import the functions:
# from twotie14 import network_loss_fitness as rpl_func
# from twotie142 import network_loss as rpl2_func

# # ----------------------------------------------------------------------
# # 2. TIME VECTOR (96 × 15-min)
# # ----------------------------------------------------------------------
# TIME_SLOTS = np.array([
#     0, 0.15, 0.30, 0.45, 1.00, 1.15, 1.30, 1.45, 2.00, 2.15, 2.30, 2.45,
#     3.00, 3.15, 3.30, 3.45, 4.00, 4.15, 4.30, 4.45, 5.00, 5.15, 5.30, 5.45,
#     6.00, 6.15, 6.30, 6.45, 7.00, 7.15, 7.30, 7.45, 8.00, 8.15, 8.30, 8.45,
#     9.00, 9.15, 9.30, 9.45, 10.00, 10.15, 10.30, 10.45, 11.00, 11.15, 11.30, 11.45,
#     12.00, 12.15, 12.30, 12.45, 13.00, 13.15, 13.30, 13.45, 14.00, 14.15, 14.30, 14.45,
#     15.00, 15.15, 15.30, 15.45, 16.00, 16.15, 16.30, 16.45, 17.00, 17.15, 17.30, 17.45,
#     18.00, 18.15, 18.30, 18.45, 19.00, 19.15, 19.30, 19.45, 20.00, 20.15, 20.30, 20.45,
#     21.00, 21.15, 21.30, 21.45, 22.00, 22.15, 22.30, 22.45, 23.00, 23.15, 23.30, 23.45
# ])

# D = len(TIME_SLOTS)          # 96
# NP = 20                      # population size

# # ----------------------------------------------------------------------
# # 3. EV PARAMETERS (same as original MATLAB)
# # ----------------------------------------------------------------------
# BC       = 50 / 4.0          # kWh per slot
# CHG_EFF  = 0.90
# CCR      = 4.0 / 4.0         # kWh per 15-min when charging
# DCR      = 1.0 / 4.0         # kWh per 15-min when discharging
# CCC      = 7.0               # $/kWh
# DCC      = 3.0

# # ----------------------------------------------------------------------
# # 4. DUMMY EV DATA (replace with real .mat if you have it)
# # ----------------------------------------------------------------------
# def generate_ev_data():
#     np.random.seed(42)
#     data = np.zeros((NP, 4))
#     data[:, 0] = np.random.uniform(6, 10, NP)   # arrival
#     data[:, 1] = np.random.uniform(16, 20, NP)  # departure
#     data[:, 2] = np.random.uniform(0.2, 0.6, NP)# SOC arrival
#     data[:, 3] = np.random.uniform(0.8, 1.0, NP)# SOC departure
#     return data

# ev_data = generate_ev_data()

# # ----------------------------------------------------------------------
# # 5. COST FUNCTION (uses rpl / rpl2 exactly as MATLAB)
# # ----------------------------------------------------------------------
# def evaluate_individual(schedule: np.ndarray, ev_idx: int) -> float:
#     """
#     schedule : (96,)  →  -1 / 0 / 1
#     ev_idx   : 0..19 (which EV in ev_data)
#     Returns  : cost (float)  – same logic as twotie14 / twotie142
#     """
#     arr_t  = ev_data[ev_idx, 0]
#     dep_t  = ev_data[ev_idx, 1]
#     soc_arr = ev_data[ev_idx, 2]
#     soc_dep = ev_data[ev_idx, 3]

#     # required energy
#     soc_diff = max(soc_dep - soc_arr, 0)
#     eng_req  = (soc_diff * BC) / CHG_EFF
#     if eng_req <= 0:
#         return 0.0

#     # find indices
#     arr_idx = np.searchsorted(TIME_SLOTS, arr_t, side='left')
#     dep_idx = np.searchsorted(TIME_SLOTS, dep_t, side='right')
#     arr_idx = max(arr_idx, 0)
#     dep_idx = min(dep_idx, D)

#     cost = 0.0
#     remaining = eng_req

#     for t in range(arr_idx, dep_idx):
#         action = schedule[t]
#         if action == 1 and remaining > 0:          # charge
#             charge = CCR
#             cost += CCC * charge
#             remaining -= charge
#             if remaining < 0:
#                 remaining = 0
#         elif action == -1 and remaining > 0:        # discharge
#             discharge = DCR
#             cost -= (DCC + discharge)               # revenue
#             remaining += discharge

#     # huge penalty if demand not satisfied
#     return cost if remaining <= 0 else cost + 1e6

# # ----------------------------------------------------------------------
# # 6. INITIAL POPULATION (random -1/0/1)
# # ----------------------------------------------------------------------
# pop = np.random.choice([-1, 0, 1], size=(NP, D))
# costs = np.array([evaluate_individual(pop[i], i) for i in range(NP)])

# best_idx = costs.argmin()
# best_cost = costs[best_idx]
# best_mem = pop[best_idx].copy()

# # ----------------------------------------------------------------------
# # 7. DE PARAMETERS (exact match with MATLAB)
# # ----------------------------------------------------------------------
# F_weight = 0.50
# F_CR     = 0.60
# MAX_ITER = 100
# STRATEGY = 5                     # DE/rand/1 with per-generation dither

# # ----------------------------------------------------------------------
# # 8. PRINT HELPERS
# # ----------------------------------------------------------------------
# def print_iteration(gen, cost, schedule):
#     line = f"{gen:6d} {cost:15.6f}"
#     for v in schedule:
#         line += f" {v: .1f}"
#     print(line)

# def print_population_matrix(population):
#     print("\n=== 96 × 20 POPULATION MATRIX (rows=time, cols=EV) ===")
#     print("Time   " + " ".join([f"EV{i+1:3d}" for i in range(NP)]))
#     for t in range(D):
#         row = f"{TIME_SLOTS[t]:5.2f} " + " ".join(f"{x:3d}" for x in population[:, t])
#         print(row)

# # ----------------------------------------------------------------------
# # 9. MAIN DE LOOP
# # ----------------------------------------------------------------------
# print(f"{'Iter':>6}  {'Best Cost':>15}  {'Schedule (first 96 values)':>40}")
# print("-" * 120)

# start = time.time()
# for gen in range(1, MAX_ITER + 1):
#     old_pop = pop.copy()

#     for i in range(NP):
#         # ---- three distinct random vectors ---------------------------------
#         idxs = [j for j in range(NP) if j != i]
#         a, b, c = old_pop[np.random.choice(idxs, 3, replace=False)]

#         # ---- mutation (strategy 5) ----------------------------------------
#         f1 = (1 - F_weight) * np.random.rand() + F_weight
#         mutant = c + f1 * (a - b)

#         # ---- crossover ----------------------------------------------------
#         cross = np.random.rand(D) < F_CR
#         if not np.any(cross):
#             cross[np.random.randint(D)] = True
#         trial = np.where(cross, mutant, old_pop[i])

#         # ---- bounce-back --------------------------------------------------
#         origin = c
#         for j in range(D):
#             if trial[j] > 1:
#                 trial[j] = 1 + np.random.rand() * (origin[j] - 1)
#             if trial[j] < -1:
#                 trial[j] = -1 + np.random.rand() * (origin[j] + 1)
#         trial = np.clip(trial, -1, 1)

#         # ---- force integer (-1,0,1) ---------------------------------------
#         trial = np.rint(trial).astype(int)

#         # ---- evaluate ------------------------------------------------------
#         trial_cost = evaluate_individual(trial, i)

#         # ---- selection ----------------------------------------------------
#         if trial_cost <= costs[i]:
#             pop[i] = trial
#             costs[i] = trial_cost
#             if trial_cost < best_cost:
#                 best_cost = trial_cost
#                 best_mem = trial.copy()

#     # ---- print iteration ---------------------------------------------------
#     print_iteration(gen, best_cost, best_mem)

# # ----------------------------------------------------------------------
# # 10. FINAL RESULTS
# # ----------------------------------------------------------------------
# elapsed = time.time() - start
# print("-" * 120)
# print(f"Optimization finished in {elapsed:.2f} s")
# print(f"Best cost : {best_cost:.6f}")

# # ---- print full 96×20 matrix (population) -------------------------------
# print_population_matrix(pop)

# # ---- save ----------------------------------------------------------------
# np.save("pop_96x20.npy", pop)
# df = pd.DataFrame(pop.T, index=[f"t={h:5.2f}" for h in TIME_SLOTS],
#                   columns=[f"EV{i+1}" for i in range(NP)])
# df.to_csv("pop_96x20.csv")
# print("\nSaved:")
# print("   pop_96x20.npy")
# print("   pop_96x20.csv")

# # ---- optional plot of best schedule --------------------------------------
# plt.figure(figsize=(12,6))
# plt.imshow(best_mem.reshape(1,-1), aspect='auto', cmap='RdYlGn',
#            vmin=-1, vmax=1, extent=[0,24,0,1])
# plt.colorbar(label='Action (-1=discharge, 0=idle, 1=charge)')
# plt.title('Best EV Schedule (96 slots)')
# plt.xlabel('Time (h)')
# plt.yticks([])
# plt.tight_layout()
# plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DE for EV charging (96 time-slots, 20 individuals)
Now includes: run_optimization() function
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. IMPORT THE COST FUNCTIONS
# ----------------------------------------------------------------------
from twotie14 import network_loss_fitness as rpl
from twotie142 import network_loss as rpl2

# ----------------------------------------------------------------------
# 2. TIME VECTOR (96 × 15-min)
# ----------------------------------------------------------------------
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

D = len(TIME_SLOTS)          # 96
NP = 20                      # population size

# ----------------------------------------------------------------------
# 3. EV PARAMETERS
# ----------------------------------------------------------------------
BC       = 50 / 4.0
CHG_EFF  = 0.90
CCR      = 4.0 / 4.0
DCR      = 1.0 / 4.0
CCC      = 7.0
DCC      = 3.0

# ----------------------------------------------------------------------
# 4. DUMMY EV DATA
# ----------------------------------------------------------------------
def generate_ev_data():
    np.random.seed(42)
    data = np.zeros((NP, 4))
    data[:, 0] = np.random.uniform(6, 10, NP)   # arrival
    data[:, 1] = np.random.uniform(16, 20, NP)  # departure
    data[:, 2] = np.random.uniform(0.2, 0.6, NP)# SOC arrival
    data[:, 3] = np.random.uniform(0.8, 1.0, NP)# SOC departure
    return data

ev_data = generate_ev_data()

# ----------------------------------------------------------------------
# 5. COST FUNCTION
# ----------------------------------------------------------------------
def evaluate_individual(schedule: np.ndarray, ev_idx: int) -> float:
    arr_t  = ev_data[ev_idx, 0]
    dep_t  = ev_data[ev_idx, 1]
    soc_arr = ev_data[ev_idx, 2]
    soc_dep = ev_data[ev_idx, 3]

    soc_diff = max(soc_dep - soc_arr, 0)
    eng_req  = (soc_diff * BC) / CHG_EFF
    if eng_req <= 0:
        return 0.0

    arr_idx = np.searchsorted(TIME_SLOTS, arr_t, side='left')
    dep_idx = np.searchsorted(TIME_SLOTS, dep_t, side='right')
    arr_idx = max(arr_idx, 0)
    dep_idx = min(dep_idx, D)

    cost = 0.0
    remaining = eng_req

    for t in range(arr_idx, dep_idx):
        action = schedule[t]
        if action == 1 and remaining > 0:
            charge = CCR
            cost += CCC * charge
            remaining -= charge
            if remaining < 0:
                remaining = 0
        elif action == -1 and remaining > 0:
            discharge = DCR
            cost -= (DCC + discharge)
            remaining += discharge

    return cost if remaining <= 0 else cost + 1e6

# ----------------------------------------------------------------------
# 6. PRINT HELPERS
# ----------------------------------------------------------------------
def print_population_matrix(population):
    print("\n=== 96 × 20 POPULATION MATRIX (rows=time, cols=EV) ===")
    print("Time   " + " ".join([f"EV{i+1:3d}" for i in range(NP)]))
    for t in range(D):
        row = f"{TIME_SLOTS[t]:5.2f} " + " ".join(f"{x:3d}" for x in population[:, t])
        print(row)

# ----------------------------------------------------------------------
# 7. run_optimization() – THIS IS WHAT main.py CALLS
# ----------------------------------------------------------------------
def run_optimization():
    """
    Runs full DE optimization.
    Returns:
        best_schedule (96,) – best individual
        best_cost (float)
        pop_matrix (96, 20) – full population
    """
    # Initial population
    pop = np.random.choice([-1, 0, 1], size=(NP, D))
    costs = np.array([evaluate_individual(pop[i], i) for i in range(NP)])

    best_idx = costs.argmin()
    best_cost = costs[best_idx]
    best_mem = pop[best_idx].copy()

    F_weight = 0.50
    F_CR     = 0.60
    MAX_ITER = 100

    start = time.time()

    for gen in range(1, MAX_ITER + 1):
        old_pop = pop.copy()

        for i in range(NP):
            idxs = [j for j in range(NP) if j != i]
            a, b, c = old_pop[np.random.choice(idxs, 3, replace=False)]

            f1 = (1 - F_weight) * np.random.rand() + F_weight
            mutant = c + f1 * (a - b)

            cross = np.random.rand(D) < F_CR
            if not np.any(cross):
                cross[np.random.randint(D)] = True
            trial = np.where(cross, mutant, old_pop[i])

            for j in range(D):
                if trial[j] > 1:
                    trial[j] = 1 + np.random.rand() * (c[j] - 1)
                if trial[j] < -1:
                    trial[j] = -1 + np.random.rand() * (c[j] + 1)
            trial = np.clip(trial, -1, 1)
            trial = np.rint(trial).astype(int)

            trial_cost = evaluate_individual(trial, i)
            if trial_cost <= costs[i]:
                pop[i] = trial
                costs[i] = trial_cost
                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_mem = trial.copy()

    elapsed = time.time() - start
    print(f"Optimization finished in {elapsed:.2f} s")
    print(f"Best cost: {best_cost:.6f}")

    # Print 96×20 matrix
    print_population_matrix(pop)

    # Return for Streamlit
    pop_matrix = pop.T  # 96 rows × 20 cols
    return best_mem, float(best_cost), pop_matrix


# ----------------------------------------------------------------------
# 8. Run only if executed directly (not imported)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    best_schedule, best_cost, pop_matrix = run_optimization()
    # Save locally
    np.save("pop_96x20.npy", pop_matrix.T)
    df = pd.DataFrame(pop_matrix, index=[f"t={h:5.2f}" for h in TIME_SLOTS],
                      columns=[f"EV{i+1}" for i in range(NP)])
    df.to_csv("pop_96x20.csv")
    print("Saved: pop_96x20.npy and pop_96x20.csv")

    # Plot
    plt.figure(figsize=(12,6))
    plt.imshow(best_schedule.reshape(1,-1), aspect='auto', cmap='RdYlGn',
               vmin=-1, vmax=1, extent=[0,24,0,1])
    plt.colorbar(label='Action')
    plt.title('Best EV Schedule')
    plt.xlabel('Time (h)')
    plt.yticks([])
    plt.tight_layout()
    plt.show()