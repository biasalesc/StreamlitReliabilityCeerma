import streamlit as st
import numpy as np
import pandas as pd

def palmgren_miner_linear_damage(rated_life, time_at_stress, stress):
    """
    Modification from the original function from reliability library  to consider the streamlit library.

    Uses the Palmgren-Miner linear damage hypothesis to find the outputs:

    Inputs:
    - rated life - an array or list of how long the component will last at a given stress level
    - time_at_stress - an array or list of how long the component is subjected to the stress that gives the specified rated_life
    - stress - what stress the component is subjected to. Not used in the calculation but is required for printing the output.
    Ensure that the time_at_stress and rated life are in the same units as the answer will also be in those units

    Outputs:
    - Fraction of life consumed per load cycle
    - service life of the component
    - Fraction of damage caused at each stress level
    """

    life_frac = []
    for i, x in enumerate(time_at_stress):
        life_frac.append(x / rated_life[i])
    life_consumed_per_load_cycle = sum(life_frac)
    service_life = 1 / life_consumed_per_load_cycle
    damage_frac = service_life * np.array(life_frac)
    st.write("**Palmgren-Miner Linear Damage Model results:**")

    main_df = pd.DataFrame({
        "Description": [
            "Life used per load cycle (%)",
            "Service life (load cycles)"
        ],
        "Value": [
            round(life_consumed_per_load_cycle * 100, 5),
            round(service_life, 5)
        ]
    })

    st.write("""**1. Component life summary**""")
    st.dataframe(main_df.set_index("Description"), use_container_width=True)

    stress_damage_df = pd.DataFrame({
        "Stress": stress,
        "Damage Fraction (%)": [round(df * 100, 5) for df in damage_frac]
    })    

    st.write("""**2. Damage per stress level**""")
    st.dataframe(stress_damage_df.set_index("Stress"), use_container_width=True)   


def show():
    st.write("""
    In this module, you can provide your data to the
    Palmgren-Miner linear damage model for fatigue analysis.
    """)
    with st.expander('Short Guide'):
        st.write("""
        In simple terms, the Palmgren Miner linear damage model is considered in cases
        of cyclic loads with variable magnitudes over time.

        """)
        st.latex(r"""
        \frac{n_1}{N_{f1}} + \frac{n_2}{N_{f2}} + \cdots = \sum \frac{n_i}{N_{fi}} \geq 1
        """)

        st.latex(r"""
        \begin{array}{ll}
        n_i &: \text{number of cycles of operation at stress level } i \\
        N_{fi} &: \text{the number of cycles at constant stress level } i \\
        \frac{n_i}{N_{fi}} &: \text{fraction of accumulated damage for the level } i \\
        \sum \frac{n_i}{N_{fi}} &: \text{total damage accumulated over all levels} \\
        \end{array}
        """)

    with st.expander('Data format'):
        st.info("""
        Upload an excel file that contains the following columns:
        * Rated_life - How long the component will last at a given stress level;
        * Time - How long the component is subjected to the stress that gives the specified Rated_life;
        * Stress - What stress the component is subjected to.
        """)
        df_show = {
            'Rated_life': [5000,6500,1000],
            'Time':[40/60,15/60,5/60],
            'Stress': [1,2,4],
        }
        df_show = pd.DataFrame.from_dict(df_show)
        st.write(df_show)


    header = st.checkbox("Does your data contain header?", value=True)
    head = 0 if header else None

    col2_1, col2_2 = st.columns(2)
    uploaded_file = col2_1.file_uploader("Upload a XLSX file",
                                         type="xlsx",
                                         accept_multiple_files=False,
                                         label_visibility="collapsed")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=head)
        col2_2.dataframe(df, use_container_width=True)

        rated_life = df.iloc[:, 0].values
        time_at_stress = df.iloc[:, 1].values
        stress = df.iloc[:, 2].values

        total_time = df.iloc[:, 1].sum()
        time_at_stress = time_at_stress / total_time

        if len(rated_life) != len(time_at_stress) or len(rated_life) != len(stress):
            st.warning("All inputs must be of equal length.")
        else: 
            if st.button("Show results"):
                palmgren_miner_linear_damage(rated_life, time_at_stress, stress)

