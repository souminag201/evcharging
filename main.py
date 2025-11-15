import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scropt import run_optimization

st.set_page_config(page_title="EV Optimizer", layout="wide")
st.title("EV Charging + Network Reconfiguration")
st.write("20 EVs × 96 Time Slots | IEEE 33-Bus | DE Optimization")

if st.button("Run Optimization (100 iterations)", type="primary"):
    with st.spinner("Optimizing..."):
        best_schedule, best_cost, population = run_optimization()

    st.success(f"Best Cost: **{best_cost:.2f}**")

    # Plot best schedule
    fig, ax = plt.subplots(figsize=(14, 3))
    img = ax.imshow(best_schedule.reshape(1, -1), cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(np.linspace(0, 95, 13))
    ax.set_xticklabels([f"{h:.0f}h" for h in range(0, 25, 2)])
    ax.set_yticks([])
    ax.set_xlabel("Time")
    plt.colorbar(img, ax=ax, label="Action")
    st.pyplot(fig)

    # Show population matrix
    with st.expander("View Full 96×20 Population Matrix"):
        df = pd.DataFrame(population.T, columns=[f"EV{i+1}" for i in range(20)])
        st.dataframe(df.style.background_gradient(cmap='RdYlGn'))

    # Download
    csv = df.to_csv(index=False)
    st.download_button("Download Population CSV", csv, "population_96x20.csv")