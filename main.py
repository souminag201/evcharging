# main.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scropt import run_optimization  # ← returns (96,), float, (96,20)

st.set_page_config(page_title="EV Optimizer", layout="wide")
st.title("EV Charging + Network Reconfiguration")
st.caption("20 EVs × 96 Time Slots (15-min intervals)")

if st.button("Run Optimization (100 iterations)", type="primary"):
    with st.spinner("Running DE optimization..."):
        best_schedule, best_cost, pop_matrix = run_optimization()  # pop_matrix: (96, 20)

    st.success(f"**Best Cost:** ${best_cost:.2f}")

    # === Plot Best Schedule ===
    fig, ax = plt.subplots(figsize=(14, 2))
    img = ax.imshow(best_schedule.reshape(1, -1), cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.linspace(0, 95, 13))
    ax.set_xticklabels([f"{h:.0f}h" for h in range(0, 25, 2)])
    ax.set_yticks([])
    ax.set_xlabel("Time (h)")
    plt.colorbar(img, ax=ax, label="‑1=discharge, 0=idle, 1=charge")
    st.pyplot(fig)

    # === 96×20 Matrix: pop_matrix is already (96, 20) ===
    with st.expander("View Full 96×20 Population Matrix (Time × EV)"):
        df = pd.DataFrame(
            pop_matrix,  # ← (96, 20) ← CORRECT
            index=[f"{h:5.2f}h" for h in np.arange(0, 24, 0.25)[:96]],  # 96 rows
            columns=[f"EV{i+1}" for i in range(20)]                     # 20 cols
        )
        st.dataframe(df.style.background_gradient(cmap="RdYlGn", axis=None))

    # === Download CSV ===
    csv = df.to_csv().encode()
    st.download_button(
        label="Download 96×20 Matrix",
        data=csv,
        file_name="ev_population_96x20.csv",
        mime="text/csv"
    )