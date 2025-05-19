import streamlit as st
import numpy as np
import pandas as pd

from reliability.PoF import acceleration_factor

def show():
    st.write("""
    In this module, the Acceleration factor is calculated using the Arrhenius equation.
    """)

    with st.expander('Short Guide'):
        st.write("""
        In simple terms, the Arrhenius model for Acceleration factor determines the relationship between 
        temperature and reaction rate.
        """)

        st.latex(r"""
                 AF = \exp\left[\frac{E_a}{K_B}\left(\frac{1}{T_{use}} - \frac{1}{T_{acc}}\right)\right]
        """)

        st.latex(r"""
        \begin{array}{ll}
        AF: \text{Acceleration factor } \\
        E_a : \text{Activation energy (eV)} \\
        T_{use}: \text{Temperature of usage}\\
        T_{acc}: \text{Temperature of acceleration}\\       
        \end{array}
        """)

    var = st.radio('Which of the following variables do you want to determine:', ('Acceleration factor', 
                                                                                  'Temperature of acceleration',
                                                                                  "Activation energy"
                                                                            ))
    
    if var == 'Acceleration factor':
        cols = st.columns([1])
        AF=None
        T_use = cols[0].number_input('The temperature of usage (Celsius)',
                                     min_value=0.0, value=0.0, step=.001, format='%0.3f')
        T_acc = cols[0].number_input('The temperature of acceleration (Celsius)',
                                     min_value=0.0, value=0.0, step=.001, format='%0.3f')
        Ea = cols[0].number_input('The activation energy (eV)',
                                  min_value=0.0, value=0.0, step=.001, format='%0.3f')

    if var == 'Temperature of acceleration':
        cols = st.columns([1])
        T_use = cols[0].number_input('The temperature of usage (Celsius)',
                                     min_value=0.0, value=0.0, step=.001, format='%0.3f')
        AF = cols[0].number_input('Acceleration factor',
                                  min_value=0.0, value=0.0, step=.001, format='%0.3f')
        T_acc = None
        Ea = cols[0].number_input('The activation energy (eV)',
                                  min_value=0.0, value=0.0, step=.001, format='%0.3f')

    if var == 'Activation energy':
        cols = st.columns([1])
        T_use = cols[0].number_input('The temperature of usage (Celsius)',
                                     min_value=0.0, value=0.0, step=.001, format='%0.3f')
        AF = cols[0].number_input('Acceleration factor',
                                  min_value=0.0, value=0.0, step=.001, format='%0.3f')
        T_acc = cols[0].number_input('The temperature of acceleration (Celsius)',
                                     min_value=0.0, value=0.0, step=.001, format='%0.3f')
        Ea = None

    if st.button('Show results'):
        af_results = acceleration_factor(AF, T_use, T_acc, Ea, print_results=False)

        st.write(f"""
                 **Results from Acceleration factor:**
                 """)
    
        results_AF = {
            "Description": [
                "Acceleration Factor",
                "Use Temperature",
                "Accelerated Temperature",
                "Activation Energy"],
            
            "Value": [
                f"{af_results.AF}",
                f"{af_results.T_use} °C",
                f"{af_results.T_acc} °C",
                f"{af_results.Ea}"
            ]}
        
        df_results = pd.DataFrame(results_AF)
        st.dataframe(df_results.set_index("Description"), use_container_width=True)