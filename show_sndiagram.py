import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as ss
import matplotlib.pyplot as plt

# from reliability.PoF import SN_diagram

def SN_diagram(
    stress,
    cycles,
    stress_runout=None,
    cycles_runout=None,
    xscale="log",
    stress_trace=None,
    cycles_trace=None,
    show_endurance_limit=None,
    method_for_bounds="statistical",
    CI=0.95,
    **kwargs
):
    """
    Modification from the original function from reliability library  to consider the plotly library and the streamlit library.

    This function will plot the stress vs number of cycles (S-N) diagram when supplied with data from a series of fatigue tests.

    Inputs:
    stress - an array or list of stress values at failure
    cycles - an array or list of cycles values at failure
    stress_runout - an array or list of stress values that did not result in failure Optional
    cycles_runout - an array or list of cycles values that did not result in failure. Optional
    xscale - 'log' or 'linear'. Default is 'log'.
    stress_trace - an array or list of stress values to be traced across to cycles values.
    cycles_trace - an array or list of cycles values to be traced across to stress values.
    show_endurance_limit - This will adjust all lines of best fit to be greater than or equal to the average stress_runout. Defaults to False if stress_runout is not specified. Defaults to True if stress_runout is specified.
    method_for_bounds - 'statistical', 'residual', or None. Defaults to 'statistical'. If set to 'statistical' the CI value is used, otherwise it is not used for the 'residual' method. Residual uses the maximum residual datapoint for symmetric bounds. Setting the method for bounds to None will turn off the confidence bounds.
    CI - Must be between 0 and 1. Default is 0.95 for 95% confidence interval. Only used if method_for_bounds = 'statistical'
    Other plotting keywords (eg. color, linestyle, etc) are accepted for the line of best fit.

    Outputs:
    The plot is the only output. All calculated values are shown on the plot.

    Example usage:
    stress = [340, 300, 290, 275, 260, 255, 250, 235, 230, 220, 215, 210]
    cycles = [15000, 24000, 36000, 80000, 177000, 162000, 301000, 290000, 361000, 881000, 1300000, 2500000]
    stress_runout = [210, 210, 205, 205, 205]
    cycles_runout = [10 ** 7, 10 ** 7, 10 ** 7, 10 ** 7, 10 ** 7]
    SN_diagram(stress=stress, cycles=cycles, stress_runout=stress_runout, cycles_runout=cycles_runout,method_for_bounds='residual',cycles_trace=[5 * 10 ** 5], stress_trace=[260])
    plt.show()
    """

    # error checking of input and changing inputs to arrays
    if type(stress) == np.ndarray:
        pass
    elif type(stress) == list:
        stress = np.array(stress)
    else:
        st.error("stress must be an array or list")
    if type(cycles) == np.ndarray:
        pass
    elif type(cycles) == list:
        cycles = np.array(cycles)
    else:
        st.error("cycles must be an array or list")
    if len(cycles) != len(stress):
        st.error("the number of datapoints for stress and cycles must be equal")

    if stress_runout is not None and cycles_runout is not None:
        if len(cycles_runout) != len(stress_runout):
            st.error(
                "the number of datapoints for stress_runout and cycles_runout must be equal"
            )
        if type(stress_runout) == np.ndarray:
            pass
        elif type(stress_runout) == list:
            stress_runout = np.array(stress_runout)
        else:
            st.error("stress_runout must be an array or list")
        if type(cycles_runout) == np.ndarray:
            pass
        elif type(cycles_runout) == list:
            cycles_runout = np.array(cycles_runout)
        else:
            st.error("cycles_runout must be an array or list")

    if method_for_bounds not in ["statistical", "residual", None]:
        st.error(
            "method_for_bounds must be either "
            "statistical"
            ","
            "residual"
            ",or None (for no bounds)."
        )

    if CI <= 0 or CI >= 1:
        st.error(
            "CI must be between 0 and 1. Default is 0.95 for 95% Confidence intervals on statistical bounds"
        )

    if stress_runout is None and show_endurance_limit is None:
        show_endurance_limit = False
    elif stress_runout is None and show_endurance_limit is True:
        st.warning(
            "Unable to show endurance limit without entries for stress_runout and cycles_runout. show_endurance_limit has been changed to False."
        )
        show_endurance_limit = False
    elif stress_runout is not None and show_endurance_limit is None:
        show_endurance_limit = True

    if xscale not in ["log", "linear"]:
        st.error(
            "xscale must be " "log" " or " "linear" ". Default is " "log" ""
        )

    if stress_trace is not None:
        if type(stress_trace) not in [np.ndarray, list]:
            st.error("stress_trace must be an array or list. Default is None")
    if cycles_trace is not None:
        if type(cycles_trace) not in [np.ndarray, list]:
            st.error("cycles_trace must be an array or list. Default is None")

    # fit the log-linear model
    log10_cycles = np.log10(cycles)
    linear_fit = np.polyfit(log10_cycles, stress, deg=1)
    m = linear_fit[0]
    c = linear_fit[1]
    xvals = np.logspace(0, max(log10_cycles) + 2, 1000)
    y = m * np.log10(xvals) + c
    y_pred = m * np.log10(cycles) + c
    residual = max(
        abs(y_pred - stress)
    )  # largest difference between line of best fit and observed data.
    # this is for the plotting limits
    cycles_min_log = 10 ** (int(np.floor(np.log10(min(cycles)))))
    cycles_max_log = 10 ** (int(np.ceil(np.log10(max(cycles)))))
    ymax = max(stress) * 1.2
    ymin = min(stress) - max(stress) * 0.2

    # extract keyword arguments
    if "label" in kwargs:
        label = kwargs.pop("label")
    else:
        label = f"σₐ = {str(round(c, 3))} - {str(round(m * -1, 3))} × log₁₀(Nᶠ)"

    if "color" in kwargs:
        color = kwargs.pop("color")
    else:
        color = "steelblue"

    if show_endurance_limit is True:
        endurance_limit = np.average(
            stress_runout
        )  # endurance limit is predicted as the average of the runout values
        y[y < endurance_limit] = endurance_limit
        y_upper = m * np.log10(xvals) + c + residual
        y_lower = m * np.log10(xvals) + c - residual
        y_upper[y_upper < endurance_limit + residual] = endurance_limit + residual
        y_lower[y_lower < endurance_limit - residual] = endurance_limit - residual
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=stress,
            mode="markers",
            marker=dict(size=5, color="black"),
            name="Failure data",
        )
    )   

    fig.add_trace(
        go.Scatter(
            x=xvals,
            y=y,
            mode="lines",
            line=dict(color=color, width=2),
            name=label,
        )
    )

    # # plot the data and lines of best fit
    # plt.scatter(cycles, stress, marker=".", color="k", label="Failure data")
    # plt.plot(xvals, y, label=label, color=color)

    if (
        show_endurance_limit is True
    ):  # this is separated from the other endurance limit calculations due to the order of entries in the legend
        fig.add_trace(
            go.Scatter(
                x=[0, max(np.hstack([cycles, cycles_runout])) * 10],
                y=[endurance_limit, endurance_limit],
                mode="lines",
                line=dict(color="orange", width=1, dash="dash"),
                name=str("Endurance limit = " + str(round(endurance_limit, 2))),
            )
        )
        # plt.plot(
        #     [0, max(np.hstack([cycles, cycles_runout])) * 10],
        #     [endurance_limit, endurance_limit],
        #     linestyle="--",
        #     color="orange",
        #     linewidth=1,
        #     label=str("Endurance limit = " + str(round(endurance_limit, 2))),
        # )

    # trace the fatigue life at the specified stresses
    if stress_trace is not None:
        for stress_value in stress_trace:
            fatigue_life = 10 ** ((stress_value - c) / m)
            fig.add_trace(
                go.Scatter(
                    x=[0, fatigue_life, fatigue_life],
                    y=[stress_value, stress_value, 0],
                    mode="lines",
                    line=dict(color="red", width=1),
                    showlegend=False,
                )
            )
            # plt.plot(
            #     [0, fatigue_life, fatigue_life],
            #     [stress_value, stress_value, 0],
            #     "r",
            #     linewidth=0.5,
            # )
            if xscale == "log":

                fig.add_annotation(
                    x=np.log10(cycles_min_log),
                    y=stress_value * 1.02,
                    text=str(round(stress_value, 2)),
                    showarrow=False,
                )
                fig.add_annotation(
                    x=np.log10(fatigue_life),
                    y=ymin,
                    text=str(int(fatigue_life)),
                    showarrow=False,
                )
            else:

                fig.add_annotation(
                    x=cycles_min_log - 1,
                    y=stress_value * 1.02,
                    text=str(round(stress_value, 2)),
                    showarrow=False,
                )
                fig.add_annotation(
                    x=fatigue_life,
                    y=ymin,
                    text=str(int(fatigue_life)),
                    showarrow=False,
                )

            # plt.text(cycles_min_log - 1, stress_value, str(stress_value))
            # plt.text(fatigue_life, ymin, str(int(fatigue_life)))

    # trace the fatigue life at the specified cycles
    if cycles_trace is not None:
        for cycles_value in cycles_trace:
            fatigue_strength = m * np.log10(cycles_value) + c
            fig.add_trace(
                go.Scatter(
                    x=[cycles_value, cycles_value, 0],
                    y=[0, fatigue_strength, fatigue_strength],
                    mode="lines",
                    line=dict(color="blue", width=1),
                    showlegend=False,
                )
            )

            # plt.plot(
            #     [cycles_value, cycles_value, 0],
            #     [0, fatigue_strength, fatigue_strength],
            #     "b",
            #     linewidth=0.5,
            # )
            if xscale == "log":
                fig.add_annotation(
                    x=np.log10(cycles_min_log),
                    y=fatigue_strength * 1.02,
                    text=str(round(fatigue_strength, 2)),
                    showarrow=False,
                )
                fig.add_annotation(
                    x=np.log10(cycles_value),
                    y=ymin,
                    text=str(int(cycles_value)),
                    showarrow=False,
                )
            else: 
                fig.add_annotation(
                    x=cycles_min_log - 1,
                    y=fatigue_strength * 1.02,
                    text=str(round(fatigue_strength, 2)),
                    showarrow=False,
                )

                fig.add_annotation(
                    x=cycles_value,
                    y=ymin,
                    text=str(int(cycles_value)),
                    showarrow=False,
                )

            # plt.text(
            #     cycles_min_log - 1, fatigue_strength, str(round(fatigue_strength, 2))
            # )
            # plt.text(cycles_value, ymin, str(int(cycles_value)))

    # set the plot limits and plot the runout data (the position for the runout data depends on the plotting limits)
    fig.update_layout(
        title=r"S-N diagram",
        xaxis=dict(
            type="log",
            title=r"Number of cycles (Nᶠ)"
            ),
        yaxis=dict(
            title=r"Stress (σₐ)"
    ))

    # plt.gca().set_xscale(xscale)
    # plt.xlabel(r"Number of cycles $(N_f)$")
    # plt.ylabel(r"Stress $(σ_a)$")
    # plt.title("S-N diagram")

    if xscale == "log":
        fig.update_layout(
            xaxis=dict(
                type="log",
                range=[np.log10(cycles_min_log), np.log10(cycles_max_log)]
                ))
        
        # plt.xlim([cycles_min_log, cycles_max_log])

        if stress_runout is not None:
            fig.add_trace(
                go.Scatter(
                    x=[cycles_max_log, cycles_max_log],
                    y=stress_runout,
                    mode="markers",
                    marker=dict(size=5, color="black", symbol = "triangle-right"),
                    name="Runout data",
                )
            )

            # plt.scatter(
            #     np.ones_like(cycles_runout) * cycles_max_log,
            #     stress_runout,
            #     marker=5,
            #     color="k",
            #     label="Runout data",
            # )
    else:
        fig.update_layout(
            xaxis=dict(
                type="linear",
                range=[0, max(cycles) * 1.2]
                ))
        
        # plt.xlim([0, max(cycles) * 1.2])
        if stress_runout is not None:
            fig.add_trace(
                go.Scatter(
                    x=np.ones_like(cycles_runout) * max(cycles) * 1.2,
                    y=stress_runout,
                    mode="markers",
                    marker=dict(size=5, color="black", symbol = "triangle-right"),
                    name="Runout data",
                )
            )
            # plt.scatter(
            #     np.ones_like(cycles_runout) * max(cycles) * 1.2,
            #     stress_runout,
            #     marker=5,
            #     color="k",
            #     label="Runout data",
            # )
    fig.update_layout(
            yaxis=dict(
                type="linear",
                range=[ymin, ymax]
                ))
    
    # plt.ylim([ymin, ymax])

    # upper and lower bounds
    if method_for_bounds == "residual":
        fig.add_trace(
            go.Scatter(
                x=xvals,
                y=y_upper,
                mode="lines",
                line=dict(color=color, width=1, dash="dash"),
                name=str("Max residual bounds (±" + str(round(residual, 2)) + ")"),
            )
        )

        # plt.plot(
        #     xvals,
        #     y_upper,
        #     color=color,
        #     alpha=0.7,
        #     linestyle="--",
        #     label=str("Max residual bounds (±" + str(round(residual, 2)) + ")"),
        # )

        fig.add_trace(
            go.Scatter(
                x=xvals,
                y=y_lower,
                mode="lines",
                line=dict(color=color, width=1, dash="dash"),
                name=str("Max residual bounds (±" + str(round(residual, 2)) + ")"),
                showlegend=False,
            )
        )

        # plt.plot(xvals, y_lower, color=color, alpha=0.7, linestyle="--")
    elif method_for_bounds == "statistical":
        x_av = np.average(cycles)
        n = len(cycles)
        y_pred = m * log10_cycles + c
        STEYX = (((stress - y_pred) ** 2).sum() / (n - 2)) ** 0.5
        tinv = ss.t.ppf((CI + 1) / 2, n - 2)
        DEVSQ = ((cycles - x_av) ** 2).sum()
        CL = tinv * STEYX * (1 / n + (xvals - x_av) ** 2 / DEVSQ) ** 0.5
        y_upper_CI = y + CL
        y_lower_CI = y - CL

        fig.add_trace(
            go.Scatter(
                x=xvals,
                y=y_upper_CI,
                mode="lines",
                line=dict(color=color, width=1, dash="dash"),
                name=str("Statistical bounds ( " + str(round(CI*100, 2)) + "% CI)"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=xvals,
                y=y_lower_CI,
                mode="lines",
                line=dict(color=color, width=1, dash="dash"),
                name=str("Statistical bounds ( " + str(round(CI*100, 2)) + "% CI)"),
                showlegend=False,
            )
        )

        # plt.plot(xvals, y_lower_CI, color=color, linestyle="--", alpha=0.7)
        # plt.plot(
        #     xvals,
        #     y_upper_CI,
        #     color=color,
        #     linestyle="--",
        #     alpha=0.7,
        #     label="Statistical bounds (95% CI)",
        # )

    # fig.update_layout(
    #     margin=dict(t=80, b=80, l=80, r=50),
    #     width=900,
    #     height=600,
    # )

    # plt.legend(loc="upper right")
    # plt.subplots_adjust(
    #     top=0.9, bottom=0.135, left=0.12, right=0.93, hspace=0.2, wspace=0.2
    # )
    # st.plotly_chart(fig, use_container_width=True, xaxis=dict(type=f"{xscale}"))

    st.plotly_chart(fig, use_container_width=True, xaxis=dict(type="log"))


def show():
    st.write("""
    In this module, you can provide your data (complete or incomplete) 
    and plot the stress vs. number of cycles (S-N) diagram.
    """)

    with st.expander('Short Guide'):
        st.write("""
        In simple terms, the S-N diagram is a graphical representation of the relationship
        between the applied stress and the number of cycles to failure of a material or component.
        """)

        st.latex(r"S = a \cdot N^{-b}")
        st.latex(r"""
        \begin{array}{ll}
        S: \text{stress amplitude} \\
        N: \text{number of cycles to failure} \\
        a, b: \text{constants to be determined} \\
        \end{array}
        """)

        st.write("""
        For the S-N diagram, the stress and number of cycles to failure are 
        fitted into a slope using least squares polynomial fit.
        """)

        st.latex(r"""
                 S = m \cdot \log_{10}(N) + c
                 """)
        
        st.info("""
        The censored data is considered to the determine the 
        endurance limit of the material or comoponent, as the average
        value of the of the stress censored data. Thus, the censored 
        data does not affect the slope of the S-N curve.
        """)

    with st.expander('Data format'):
        st.info("""
        Upload an excel file that contains the following columns:
        * Type - 'F' for failure time, 'C' for right-censored time;
        * Stress - stress level associated with this failure and/or that did not result in failure;
        * Cycles - the cycles values at failure and/or that did not result in failure.
        """)
        df_show = {
            'Type': ['F','F','F','F','F','F','F','F','F','F','F','F'],
            'Stress':[340, 300, 290, 275, 260, 255, 250, 235, 230, 220, 215, 210],
            'Cycles': [15000, 24000, 36000, 80000, 177000, 162000, 301000, 290000, 361000, 881000, 1300000, 2500000],
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

        # Seleciona dados com base na posição das colunas
        type_col = df.iloc[:, 0]
        stress = df.loc[type_col == 'F', df.columns[1]].to_numpy()
        cycles = df.loc[type_col == 'F', df.columns[2]].to_numpy()

        # Checa se há runouts ('C')
        has_runout = 'C' in type_col.values
        stress_runout = cycles_runout = None
        if has_runout:
            stress_runout = df.loc[type_col == 'C', df.columns[1]].to_numpy()
            cycles_runout = df.loc[type_col == 'C', df.columns[2]].to_numpy()
    
        cols = st.columns([1, 1])
        x_scale = cols[0].radio('Choose the scale for the x_axis:', ('log', 'linear'))

        # Define argumentos comuns
        sn_kwargs = {
            'stress': stress,
            'cycles': cycles,
            'xscale': x_scale
        }

        if has_runout:
            sn_kwargs['stress_runout'] = stress_runout
            sn_kwargs['cycles_runout'] = cycles_runout

            bounds = cols[1].radio('Choose the method for the confidence bounds:',
                    ('None', 'statistical', 'residual'))
        else:
            bounds = cols[1].radio('Choose the method for the confidence bounds:',
                    ('None', 'statistical'))

        # Adiciona parâmetros de bounds se necessário
        if bounds == 'statistical':
            CI_defined = st.number_input(
                label='Define the confidence level:',
                min_value=0.01,
                max_value=0.99,
                value=0.95,
                step=0.01,
                format='%.2f'
            )
            sn_kwargs['method_for_bounds'] = bounds
            sn_kwargs['CI'] = CI_defined
        elif bounds != 'None':
            sn_kwargs['method_for_bounds'] = bounds
        
        if st.checkbox("Do you want to trace stress and cycle values on the plot?"):
            st.markdown("Enter values ​​separated by semicolons (ex: 230;220;210).")
            stress_trace_input = st.text_input("Values for stress:")
            cycles_trace_input = st.text_input("Values for cycles:")

            try:
                if stress_trace_input and cycles_trace_input:
                    stress_trace = list(map(float, stress_trace_input.split(';')))
                    cycles_trace = list(map(float, cycles_trace_input.split(';')))
                    sn_kwargs['stress_trace'] = stress_trace
                    sn_kwargs['cycles_trace'] = cycles_trace
            except ValueError:
                st.warning('Verify the input values!')

        # Plot e exibição
        if st.button("Plot S-N diagram"):
            SN_diagram(**sn_kwargs)

            # if bounds == 'statistical':
            #     legend = plt.legend()
            #     if legend:
            #         for text in legend.get_texts():
            #             if 'CI' in text.get_text():
            #                 text.set_text(f'Statistical bounds ({int(CI_defined*100)}% CI)')

            # st.pyplot(plt.gcf(), use_container_width=True)
            

