import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from reliability.PoF import fracture_mechanics_crack_initiation

class fracture_mechanics_crack_growth:
    """
    Modification from the original function from reliability library  to consider the plotly library.

    This function uses the principles of fracture mechanics to find the number of cycles required to grow a crack from an initial length until a final length.
    The final length (a_final) may be specified, but if not specified then a_final will be set as the critical crack length (a_crit) which causes failure due to rapid fracture.
    This functions performs the same calculation using two methods: similified and iterative.
    The simplified method assumes that the geometry factor (f(g)), the stress (S_net), and the critical crack length (a_crit) are constant. THis method is the way most textbooks show these problems solved as they can be done in a few steps.
    The iterative method does not make the assumptions that the simplified method does and as a result, the parameters f(g), S_net and a_crit must be recalculated based on the current crack length at every cycle.

    This function is applicable only to thin plates with an edge crack or a centre crack (which is to be specified using the parameter crack_type).
    You may also use this function for notched components by specifying the parameters Kt and D which are based on the geometry of the notch.
    For any notched components, this method assumes the notched component has a "shallow notch" where the notch depth (D) is much less than the plate width (W).
    The value of Kt for notched components may be found at https://www.efatigue.com/constantamplitude/stressconcentration/
    In the case of notched components, the local stress concentration from the notch will often cause slower crack growth.
    In these cases, the crack length is calculated in two parts (stage 1 and stage 2) which can clearly be seen on the plot using the iterative method.
    The only geometry this function is designed for is unnotched and notched thin flat plates. No centre holes are allowed.

    Inputs:
    Kc - fracture toughness
    Kt - stress concentration factor (default is 1 for no notch).
    D - depth of the notch (default is None for no notch). A notched specimen is assumed to be doubly-notched (equal notches on both sides)
    C - material constant (sometimes referred to as A)
    m - material constant (sometimes referred to as n). This value must not be 2.
    P - external load on the material (MPa)
    t - plate thickness (mm)
    W - plate width (mm)
    a_initial - initial crack length (mm) - default is 1 mm
    a_final - final crack length (mm) - default is None in which case a_final is assumed to be a_crit (length at failure). It is useful to be able to enter a_final in cases where there are multiple loading regimes over time.
    crack_type - must be either 'edge' or 'center'. Default is 'edge'. The geometry factor used for each of these in the simplified method is 1.12 for edge and 1.0 for center. The iterative method calculates these values exactly using a_initial and W (plate width).
    print_results - True/False. Default is True
    show_plot - True/False. Default is True. If True the Iterative method's crack growth will be plotted.

    Outputs:
    If print_results is True, all outputs will be printed with some description of the process.
    If show_plot is True, the crack growth plot will be shown for the iterative method.
    You may also access the following parameters from the calculated object:
    - Nf_stage_1_simplified
    - Nf_stage_2_simplified
    - Nf_total_simplified
    - final_crack_length_simplified
    - transition_length_simplified
    - Nf_stage_1_iterative
    - Nf_stage_2_iterative
    - Nf_total_iterative
    - final_crack_length_iterative
    - transition_length_iterative

    Example usage:
    fracture_mechanics_crack_growth(Kc=66,C=6.91*10**-12,m=3,P=0.15,W=100,t=5,Kt=2.41,a_initial=1,D=10,crack_type='edge')
    fracture_mechanics_crack_growth(Kc=66,C=3.81*10**-12,m=3,P=0.103,W=100,t=5,crack_type='center')
    """

    def __init__(
        self,
        Kc,
        C,
        m,
        P,
        W,
        t,
        Kt=1.0,
        a_initial=1.0,
        D=None,
        a_final=None,
        crack_type="edge",
        print_results=True,
        show_plot=True,
    ):
        if m == 2:
            raise ValueError("m can not be 2")
        if crack_type not in ["center", "edge", "centre"]:
            raise ValueError(
                "crack_type must be either edge or center. default is center"
            )
        if D is None:
            d = 0
        else:
            d = D
        if W - 2 * d < 0:
            error_str = str(
                "The specified geometry is invalid. A doubly notched specimen with specified values of the d = "
                + str(d)
                + "mm will have notches deeper than the width of the plate W = "
                + str(W)
                + "mm. This would result in a negative cross sectional area."
            )
            raise ValueError(error_str)
        # Simplified method (assuming fg, S_max, af to be constant)
        S_max = P / (t * (W - 2 * d)) * 10 ** 6
        if crack_type == "edge":
            f_g_fixed = 1.12
        elif crack_type in ["center", "centre"]:
            f_g_fixed = 1.0
        m_exp = -0.5 * m + 1
        a_crit = (
            1 / np.pi * (Kc / (S_max * f_g_fixed)) ** 2 + d / 1000
        )  # critical crack length to cause failure
        if a_final is None:
            a_f = a_crit
        elif a_final < a_crit * 1000 - d:  # this is approved early stopping
            a_f = (a_final + d) / 1000
        else:
            st.warning(
                str(
                    "WARNING: In the simplified method, the specified a_final ("
                    + str(a_final)
                    + " mm) is greater than the critical crack length to cause failure ("
                    + str(round(a_crit * 1000 - d, 5))
                    + " mm)."
                )
            )
            st.warning(
                "         a_final has been set to equal a_crit since cracks cannot grow beyond the critical length."
            )
            a_f = a_crit
        lt = (
            d / ((1.12 * Kt / f_g_fixed) ** 2 - 1) / 1000
        )  # find the transition length due to the notch
        if lt > a_initial / 1000:  # two step process due to local stress concentration
            Nf_1 = (lt ** m_exp - (a_initial / 1000) ** m_exp) / (
                m_exp * C * S_max ** m * np.pi ** (0.5 * m) * f_g_fixed ** m
            )
            a_i = lt + d / 1000  # new initial length for stage 2
        else:
            a_i = a_initial / 1000
            Nf_1 = 0
        Nf_2 = (a_f ** m_exp - a_i ** m_exp) / (
            m_exp * C * S_max ** m * np.pi ** (0.5 * m) * f_g_fixed ** m
        )
        Nf_tot = Nf_1 + Nf_2
        self.Nf_stage_1_simplified = Nf_1
        self.Nf_stage_2_simplified = Nf_2
        self.Nf_total_simplified = Nf_tot
        self.final_crack_length_simplified = a_f * 1000 - d
        self.transition_length_simplified = lt * 1000
        if print_results is True:
            st.write("""
                     **Results from fracture mechanics crack growth:**
                     """)
            # st.write("SIMPLIFIED METHOD (keeping f(g), S_max, and a_crit as constant):")
            st.write(r"**1. Simplified method (keeping $f(g)$, $S_{\text{max}}$, and $a_{\text{crit}}$ as constant):**")

            if Nf_1 == 0:
                st.write(f"Crack growth was found in a single stage since the transition length ( \
                {round(self.transition_length_simplified, 2)} mm ) was less than the initial \
                crack length {round(a_initial, 2)} mm.")
            else:
                st.write(f"Crack growth was found in two stages since the transition length ( \
                {round(self.transition_length_simplified, 2)} mm ) due to the notch, was \
                greater than the initial crack length ( {round(a_initial, 2)} mm ).")  

                st.write(f"Stage 1 (initial crack to transition length):\
                         {int(np.floor(self.Nf_stage_1_simplified))} cycles")   
                st.write(f"Stage 2 (transition length to final crack):\
                         {int(np.floor(self.Nf_stage_2_simplified))} cycles") 
                 
            if a_final is None or a_final >= a_crit * 1000 - d:
                st.write(f"Total cycles to failure: {int(np.floor(self.Nf_total_simplified))} cycles.")
                st.write(f"Critical crack length to cause failure was found to be: \
                         {round(self.final_crack_length_simplified, 2)} mm.")
            else:
                st.write(f"Total cycles to reach a_final: {int(np.floor(self.Nf_total_simplified))} cycles.")
                st.write("Note that a_final will not result in failure. To find cycles to failure, leave a_final as None.")
            st.write("")

        # Iterative method (recalculating fg, S_max, af at each iteration)
        a = a_initial
        a_effective = a_initial + d
        if crack_type in ["center", "centre"]:
            f_g = (1 / np.cos(np.pi * a_effective / W)) ** 0.5
        elif crack_type == "edge":
            f_g = (
                1.12
                - 0.231 * (a_effective / W)
                + 10.55 * (a_effective / W) ** 2
                - 21.72 * (a_effective / W) ** 3
                + 30.39 * (a_effective / W) ** 4
            )
        lt2 = d / ((1.12 * Kt / f_g) ** 2 - 1)
        self.transition_length_iterative = lt2
        self.Nf_stage_1_iterative = 0
        N = 0
        growth_finished = False
        a_array = []
        a_crit_array = []
        N_array = []
        while growth_finished is False:
            area = t * (W - 2 * d - a)
            S = (
                P / area
            ) * 10 ** 6  # local stress due to reducing cross-sectional area
            if a < lt2:  # crack growth slowed by transition length
                if crack_type in ["center", "centre"]:
                    f_g = (
                        1 / np.cos(np.pi * a / W)
                    ) ** 0.5  # Ref: p92 of Bannantine, et al. (1997).
                elif crack_type == "edge":
                    f_g = (
                        1.12
                        - 0.231 * (a / W)
                        + 10.55 * (a / W) ** 2
                        - 21.72 * (a / W) ** 3
                        + 30.39 * (a / W) ** 4
                    )  # Ref: p92 of Bannantine, et al. (1997).
                delta_K = f_g * S * (np.pi * a / 1000) ** 0.5
            else:
                if crack_type in ["center", "centre"]:
                    f_g = (1 / np.cos(np.pi * a / W)) ** 0.5
                elif crack_type == "edge":
                    f_g = (
                        1.12
                        - 0.231 * (a / W)
                        + 10.55 * (a / W) ** 2
                        - 21.72 * (a / W) ** 3
                        + 30.39 * (a / W) ** 4
                    )
                delta_K = f_g * S * (np.pi * a_effective / 1000) ** 0.5
            da = (C * delta_K ** m) * 1000
            a_crit = 1 / np.pi * (Kc / (f_g * S)) ** 2 + d / 1000
            a_crit_array.append(a_crit * 1000 - d)
            a_effective += da  # grow the crack by da
            a += da  # grow the crack by da
            N += 1
            a_array.append(a)
            N_array.append(N)
            if a_array[N - 2] < lt2 and a_array[N - 1] > lt2:
                self.Nf_stage_1_iterative = N - 1
            if a_effective > a_crit * 1000:
                growth_finished = True
            if a_final is not None:
                if a_effective > a_final + d:
                    growth_finished = True
        self.Nf_total_iterative = N
        self.final_crack_length_iterative = a_crit * 1000 - d
        self.Nf_stage_2_iterative = N - self.Nf_stage_1_iterative
        if a_final is not None:
            if a_final > a_crit * 1000 - d:
                st.warning(
                    str(
                        "WARNING: During the iterative method, the specified a_final ("
                        + str(a_final)
                        + " mm) was found to be greater than the critical crack length to cause failure ("
                        + str(round(self.final_crack_length_iterative, 2))
                        + " mm)."
                    )
                )
        if print_results is True:
            print(
                "ITERATIVE METHOD (recalculating f(g), S_max, and a_crit for each cycle):"
            )
            if a_initial > lt2:
                print(
                    "Crack growth was found in a single stage since the transition length (",
                    round(self.transition_length_iterative, 2),
                    "mm ) was less than the initial crack length",
                    round(a_initial, 2),
                    "mm.",
                )
            else:
                print(
                    "Crack growth was found in two stages since the transition length (",
                    round(self.transition_length_iterative, 2),
                    "mm ) due to the notch, was greater than the initial crack length (",
                    round(a_initial, 2),
                    "mm ).",
                )
                print(
                    "Stage 1 (a_initial to transition length):",
                    round(self.Nf_stage_1_iterative, 2),
                    "cycles",
                )
                print(
                    "Stage 2 (transition length to a_final):",
                    round(self.Nf_stage_2_iterative, 2),
                    "cycles",
                )
            if a_final is None or a_final >= a_crit * 1000 - d:
                print(
                    "Total cycles to failure:",
                    round(self.Nf_total_iterative, 2),
                    "cycles.",
                )
                print(
                    "Critical crack length to cause failure was found to be:",
                    round(self.final_crack_length_iterative, 2),
                    "mm.",
                )
            else:
                print(
                    "Total cycles to reach a_final:",
                    round(self.Nf_total_iterative, 2),
                    "cycles.",
                )
                print(
                    "Note that a_final will not result in failure. To find cycles to failure, leave a_final as None."
                )

        if show_plot is True:
            fig = go.Figure()


            # Linha do comprimento crítico da trinca
            fig.add_trace(go.Scatter(
                x=N_array,
                y=a_crit_array,
                mode='lines',
                name='Critical crack length',
                line=dict(color='darkorange')
            ))

            # Linha do comprimento da trinca ao longo dos ciclos
            fig.add_trace(go.Scatter(
                x=N_array,
                y=a_array,
                mode='lines',
                name='Crack length',
                line=dict(color='steelblue')
            ))

            # Linha tracejada preta indicando o ponto final
            fig.add_trace(go.Scatter(
                x=[0, N, N],
                y=[max(a_array), max(a_array), 0],
                mode='lines',
                name='Failure reference',
                line=dict(dash='dash', color='black', width=1),
                showlegend=False
            ))

            fig.add_annotation(
                x=N * 0.05,
                y=max(a_array),
                text=f"{round(max(a_array), 2)} mm",
                showarrow=False,
                yanchor="bottom"
            )

            fig.add_annotation(
                x=N,
                y=0,
                text=f"{int(N)} cycles",
                showarrow=False,
                xanchor="right",
                yanchor="bottom"
            )

            # Layout
            fig.update_layout(
                title="Crack growth using iterative method",
                xaxis_title="Cycles",
                yaxis_title="Crack length (mm)",
                xaxis=dict(range=[0, N * 1.1]),
                yaxis=dict(range=[0, max(a_crit_array) * 1.2]),
            )

            st.plotly_chart(fig, use_container_width=True)

def show():
    st.write("""
    In this module, fracture mechanics applied to thin plates are explored.
    Feature mechanics is an approach to fatigue analysis that involves calculating 
    the number of cycles until failure of a component that is undergoing cyclic loading.
    """)

    cols = st.columns([1])
    method = cols[0].radio('Choose the specific crack analysis:', ('Crack initiation', 'Crack growth'))

    if method == 'Crack initiation':

        with st.expander('Short Guide'):
            st.write("""
            In simple terms, the Crack initiation uses the material properties,
            the local cross-sectional area, and force applied to the component to determine 
            how many cycles until crack initiation.
            
            The mean stress correction methods considered for the crack initiation are:
            """)

            st.latex(r"""
            \text{Morrow: } \epsilon_{\text{tot}} = \frac{\Delta \epsilon}{2} = \frac{\sigma_f' - 
                     \sigma_m}{E} \left( 2 N_f \right)^b + \epsilon_f' \left( 2 N_f \right)^c
            """)

            st.latex(r"""
            \text{Modified Morrow: } \epsilon_{\text{tot}} = \frac{\sigma_f' - \sigma_m}{E} 
                     \left( 2 N_f \right)^b + \epsilon_f' \left( \frac{\sigma_f' - \sigma_m}{\sigma_f'} 
                     \right)^{\frac{c}{b}} \left( 2 N_f \right)^c
            """)

            st.latex(r"""
            \text{Smith-Watson-Topper: } \left( \sigma_m + \sigma_a \right) \frac{\Delta \epsilon}{2} = 
                     \frac{\left( \sigma_f' \right)^2}{E} \left( 2 N_f \right)^{2b} + \sigma_f' \epsilon_f' 
                     \left( 2 N_f \right)^{b+c}
            """)

            st.write("""
                    - Morrow's equation is consistent with the observation that mean stress has a significant 
                    effect at long fatigue life with low plastic strains and negligible effect on 
                    short fatigue life;
                    - Modified Morrow's equation uses the assumption that die ratio of elastic to plastic 
                    strain is dependent on mean stress;
                    - SWT method consider the maximum stress during one cycle.
                    """)
            
            st.write("""
                     """)
            
            st.info("""
                    The Morrow or Modified Morrow correction method should be used for loading sequences 
                    that are predominantly compressive, and the SWT method should be used for those that 
                    are predominantly tensile. 
                    """)

        mod = st.selectbox("Which type of mean stress correction method would you like to choose?",
                        ("Morrow", "Modified Morrow", "Smith-Watson-Topper"))
        
        dic_mod = {"Morrow": "morrow", 
                    "Modified Morrow": "modified_morrow",
                    "Smith-Watson-Topper": "SWT"}
        
        cols = st.columns([1,1])

        P = cols[0].number_input('Force applied on the component (MPa)',
            min_value=0.0, value=0.0, step=.001, format='%0.3f')
        
        A = cols[0].number_input('Cross sectional area of the component (at the point of crack initiation) (mm^2)',
            min_value=0.0, value=0.0, step=.001, format='%0.3f')
        
        Sy = cols[0].number_input('Yield strength of the material (MPa)',
            min_value=0.0, value=0.0, step=.001, format='%0.3f')
        
        E = cols[0].number_input('Elastic modulus (Young’s modulus) (MPa)',
            min_value=0.0, value=0.0, step=.001, format='%0.3f')
        
        K = cols[0].number_input('Strength coefficient of the material',
            min_value=0.0, value=0.0, step=.001, format='%0.3f')
        
        n = cols[0].number_input('Strain hardening exponent of the material',
            min_value=0.0, value=0.0, step=.001, format='%0.3f')

        b = cols[1].number_input('Elastic strain exponent of the material',
                                 value=0.0, step=.001, format='%0.3f')
        
        c = cols[1].number_input('Plastic strain exponent of the material',
                                 value=0.0, step=.001, format='%0.3f')
        
        sigma_f = cols[1].number_input('Fatigue strength coefficient of the material',
            min_value=0.0, value=0.0, step=.001, format='%0.3f')
        
        epsilon_f = cols[1].number_input('Fatigue strain coefficient of the material',
            min_value=0.0, value=0.0, step=.001, format='%0.3f')
        
        q = cols[1].number_input('Notch sensitivity factor',
            min_value=0.0, value=1.0, step=.0001, format='%0.4f',
            help= 'If un-notched, the parameter must be equal to 1')
        
        Kt = cols[1].number_input('Stress concentration factor',
            min_value=0.0, value=1.0, step=.001, format='%0.3f',
            help= 'If un-notched, the parameter must be equal to 1')
        
        if st.button("Show results"):

            fm_ci = fracture_mechanics_crack_initiation(P, A, Sy, E, K, n, b, c, 
                                            sigma_f, epsilon_f, Kt, q, 
                                            mean_stress_correction_method = dic_mod[f"{mod}"], 
                                            print_results=False)
            
            st.write("""
                     **Results from fracture mechanics crack initiation:**
                     """)

            results_crack_initiation = {
                "Description": [
                    "Cycles to form 1 mm crack",
                    "Minimum stress (MPa)",
                    "Maximum stress (MPa)",
                    "Mean stress (MPa)",
                    "Minimum strain",
                    "Maximum strain",
                    "Mean strain"
                ],
                "Value": [
                    f"{round(fm_ci.cycles_to_failure, 2)} ({round(fm_ci.cycles_to_failure * 2, 2)}) reversals",
                    round(fm_ci.sigma_min, 2),
                    round(fm_ci.sigma_max, 2),
                    round(fm_ci.sigma_mean, 2),
                    round(fm_ci.epsilon_min, 5),
                    round(fm_ci.epsilon_max, 5),
                    round(fm_ci.epsilon_mean, 5)
                ]
            }

            df_results = pd.DataFrame(results_crack_initiation)
            st.dataframe(df_results.set_index("Description"), use_container_width=True)
            
    if method == 'Crack growth':
        with st.expander('Short Guide'):
            st.write("""
            In simple terms, the life of a sctrcuture can often be separated into two stages:
                     
            - **Stage 1:** life to the crack formation of the order of 1 mm (strain-life);
            - **Stage 2:** life from crack formation to failure (fracture mechanics).
            """)

            st.write("""
                     """)
            
            st.write("""
                     The Crack growth uses the principles of fracture mechanics to find the 
                     number of cycles required to grow a crack from an initial length until 
                     a final length.

                     The equation used to estimate crack growth per load cycle is given by Paris's Law:
                     """)
            
            st.latex(r"""
                     \frac{dN}{da} = C \cdot (\Delta K)^m
                     """)
            st.latex(r"""
            \begin{array}{ll}
            \frac{dN}{da}: \text{crack growth rate per load cycle} \\
            C,m: \text{material constants (obtained experimentally)} \\
            \Delta K: \text{variation of the the stress intensity factor } K\\
            \end{array}
            """)

            st.latex(r"""
                     \Delta K = f(g) \cdot \sigma \cdot \sqrt{\pi a}
                     """)  
            st.latex(r"""
            \begin{array}{ll}
            f(g): \text{correction factor} \\
            \sigma: \text{remote stress applied to component} \\
            a: \text{crack length}\\
            \end{array}
            """)         
                     

            st.info("""
            The final length may be specified, but if not specified then the final length
            will be set as the critical crack length, which causes failure due to rapid fracture.
            """)
        
        cols = st.columns([1,1])

        P = cols[0].number_input('External load on the material (MPa)',
            min_value=0.0, value=0.0, step=.001, format='%0.3f')
        t = cols[0].number_input('Plate thickness (mm)',
            min_value=0.0, value=0.0, step=.001, format='%0.3f')
        W = cols[0].number_input('Plate width (mm)',
            min_value=0.0, value=0.0, step=.001, format='%0.3f')
        a_initial = cols[0].number_input('Initial crack length (mm)',
            min_value=0.0, value=1.0, step=.001, format='%0.3f')
        
        a_final_check = cols[0].checkbox("Do you want to specify a final crack length?")
        if a_final_check:
            a_final = cols[0].number_input('Final crack length (mm)',
                                           min_value=0.0, value=1.0, step=.001, format='%0.3f')
        else:
            a_final = None
        
        Kt = cols[1].number_input('Stress concentration factor',
            min_value=0.0, value=1.0, step=.001, format='%0.3f',
            help= 'If un-notched, the parameter must be equal to 1')
        Kc = cols[1].number_input('Fracture toughness',
            min_value=0.0, value=0.0, step=.001, format='%0.3f')

        C = cols[1].text_input('Material constant (sometimes referred to as A)',
                               value = "0.000",
                               help= f"(e.g., 6.91*10**-12)")
        try:
            C = float(eval(C))
        except:
            st.error("Invalid value for C. Please enter a valid number.")
            C = 0.0

        m = cols[1].number_input('Material constant (sometimes referred to as n)',
            min_value=0.0, value=0.0, step=.001, format='%0.3f')
        
        crack_type = cols[1].radio('Choose the specific crack type:', 
                                   ('edge', 'center'))

        if Kt == 1:
            D = None
        else:
            D = cols[1].number_input('Depth of the notch',min_value=0.0, 
                                     value=0.0, step=.001, format='%0.3f')
        
        if st.button("Show results"):
            st.write("""
                     **Results from fracture mechanics crack growth:**
                     """)

            fm_cg = fracture_mechanics_crack_growth(Kc, C, m, P, W, t, Kt, 
                                                    a_initial, D, a_final, crack_type, 
                                                    print_results=False, show_plot=False)
            
            st.write(r"**1. Simplified method (keeping $f(g)$, $S_{\text{max}}$, and $a_{\text{crit}}$ as constant):**")

            results_simplified = pd.DataFrame({
            "Description": [
                "Transition length (mm)",
                "Number of cycles in stage 1 of crack growth",
                "Number of cycles in stage 2 of crack growth",
                "Total number of cycles to failure",
                "Final crack length (mm)",
            ],

            "Value": [
                round(fm_cg.transition_length_simplified, 2),
                round(fm_cg.Nf_stage_1_simplified, 2),
                round(fm_cg.Nf_stage_2_simplified, 2),
                round(fm_cg.Nf_total_simplified, 2),
                round(fm_cg.final_crack_length_simplified, 2),
            ]})
            st.dataframe(results_simplified.set_index("Description"), use_container_width=True)

            st.write(r"**2. Iterative method (recalculating $f(g)$, $S_{\text{max}}$, and $a_{\text{crit}}$ for each cycle):**")

            results_iterative = pd.DataFrame({
            "Description": [
                "Transition length (mm)",
                "Number of cycles in stage 1 of crack growth",
                "Number of cycles in stage 2 of crack growth",
                "Total number of cycles to failure",
                "Final crack length (mm)",
            ],

            "Value": [
                round(fm_cg.transition_length_iterative, 2),
                round(fm_cg.Nf_stage_1_iterative, 2),
                round(fm_cg.Nf_stage_2_iterative, 2),
                round(fm_cg.Nf_total_iterative, 2),
                round(fm_cg.final_crack_length_iterative, 2),
            ]})
            st.dataframe(results_iterative.set_index("Description"), use_container_width=True)

            fracture_mechanics_crack_growth(Kc, C, m, P, W, t, Kt, 
                                                    a_initial, D, a_final, crack_type, 
                                                    print_results=False, show_plot=True)