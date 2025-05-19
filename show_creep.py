import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc

from reliability.PoF import creep_failure_time

def creep_rupture_curves(
    temp_array, stress_array, TTF_array, stress_trace=None, temp_trace=None
):
    """
    Modification from the original function from reliability library  to consider the plotly library and the streamlit library.

    Plots the creep rupture curves for a given set of creep data. Also fits the lines of best fit to each temperature.
    The time to failure for a given temperature can be found by specifying stress_trace and temp_trace.

    Inputs:
    temp_array: an array or list of temperatures
    stress_array: an array or list of stresses
    TTF_array: an array or list of times to failure at the given temperatures and stresses
    stress_trace: *only 1 value is accepted
    temp_trace: *only 1 value is accepted

    Outputs:
    The plot is the only output. Use plt.show() to show it.

    Example Usage:
    TEMP = [900,900,900,900,1000,1000,1000,1000,1000,1000,1000,1000,1100,1100,1100,1100,1100,1200,1200,1200,1200,1350,1350,1350]
    STRESS = [90,82,78,70,80,75,68,60,56,49,43,38,60.5,50,40,29,22,40,30,25,20,20,15,10]
    TTF = [37,975,3581,9878,7,17,213,1493,2491,5108,7390,10447,18,167,615,2220,6637,19,102,125,331,3.7,8.9,31.8]
    creep_rupture_curves(temp_array=TEMP, stress_array=STRESS, TTF_array=TTF, stress_trace=70, temp_trace=1100)
    plt.show()
    """

    if (stress_trace is not None and temp_trace is None) or (
        stress_trace is None and temp_trace is not None
    ):
        st.error(
            "You must enter both stress_trace and temp_trace to obtain the time to failure at a given stress and temperature."
        )
    if len(temp_array) < 2 or len(stress_array) < 2 or len(TTF_array) < 2:
        st.error(
            "temp_array, stress_array, and TTF_array must each have at least 2 data points for a line to be fitted."
        )
    if len(temp_array) != len(stress_array) or len(temp_array) != len(TTF_array):
        st.error(
            "The length of temp_array, stress_array, and TTF_array must all be equal"
        )

    xmin = 10 ** (int(np.floor(np.log10(min(TTF_array)))) - 1)
    xmax = 10 ** (int(np.ceil(np.log10(max(TTF_array)))) + 1)
    xvals = np.logspace(np.log10(xmin), np.log10(xmax), 100)

    delta = (max(stress_array) - min(stress_array)) * 0.2
    ymin = min(stress_array) - delta
    ymax = max(stress_array) + delta

    unique_temps = sorted(set(temp_array))
    fig = go.Figure()

    color_palette = pc.qualitative.Plotly
    temp_to_color = {T: color_palette[i % len(color_palette)] for i, T in enumerate(unique_temps)}
    temp_value = None
    for T in unique_temps:
        indices = [i for i, t in enumerate(temp_array) if t == T]
        xvalues = [TTF_array[i] for i in indices]
        yvalues = [stress_array[i] for i in indices]
        color = temp_to_color[T]

        # Scatter points
        fig.add_trace(go.Scatter(
            x=xvalues,
            y=yvalues,
            mode='markers',
            name=f'{T} °C',
            marker=dict(size=8, color=color),
        ))

        # Fit line
        fit = np.polyfit(np.log10(xvalues), yvalues, deg=1)
        m, c = fit
        yfit = m * np.log10(xvals) + c

        fig.add_trace(go.Scatter(
            x=xvals,
            y=yfit,
            mode='lines',
            line=dict(color=color, dash='solid'),
            name=f'Fit {T} °C',
            showlegend=False
        ))

        # Trace marker
        if stress_trace is not None and temp_trace is not None and T == temp_trace:
            y = stress_trace
            x = 10 ** ((y - c) / m)
            temp_value = x
            # Dashed guide lines
            fig.add_trace(go.Scatter(
                x=[xmin, x, x],
                y=[y, y, ymin],
                mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ))

    if stress_trace is not None and temp_trace is not None and temp_value is not None:
        fig.update_layout(
            xaxis=dict(type="log", title="Time to failure", range=[np.log10(xmin), np.log10(xmax)]),
            yaxis=dict(title="Stress", range=[ymin, ymax]),
            title=f"Creep Rupture Curves - For stress {stress_trace} and temperature {temp_trace}, the time to failure is {round(temp_value, 3)}",
            legend_title="Temperature"
        )
    else:
        fig.update_layout(
            xaxis=dict(type="log", title="Time to failure", range=[np.log10(xmin), np.log10(xmax)]),
            yaxis=dict(title="Stress", range=[ymin, ymax]),
            title="Creep Rupture Curves",
            legend_title="Temperature"
        )   

    st.plotly_chart(fig, use_container_width=True, xaxis=dict(type="log"))


def show():
    st.write("""
    In this module, two functions to determine time to failure due to creep can be explored. 
    Creep is the progressive accumulation of plastic strain in a component under stress at 
    an elevated temperature over a period of time.
    """)

    cols = st.columns([1])
    method = cols[0].radio('Choose the specific creep analysis:', ('Creep failure time', 
                                                                   'Creep rupture plot'))

    if method == 'Creep failure time':

        with st.expander('Short Guide'):
            st.write("""
            In simple terms, this module uses the Larson-Miller relation to find the time 
            to failure due to creep.
            """)

            st.latex(r"""
                     P = (\theta + 460)\left(C + \log_{10} t\right)
            """)

            st.latex(r"""
            \begin{array}{ll}
            P: \text{Larson-Miller parameter} \\
            \theta: \text{temperature (F) } \\
            t : \text{time in hours to rupture} \\
            C: \text{constant, usually assumed to be 20.}\\
            \end{array}
            """)

            st.write("""
                     The method uses a known failure time at a lower failure temperature to find 
                     the unknown failure time at the higher temperature.
                     """)
        

        temp_unit = st.radio('Choose the temperature unit to input:', ('Fahrenheit', 'Celsius'))
        cols = st.columns([1,1])
        temp_low = cols[0].number_input('Temperature at which the time of failure is known:',
                                        min_value=0.0, value=0.0, step=.001, format='%0.3f')
        temp_high = cols[0].number_input('Temperature at which the time of failure is unknown:',
                                        min_value=0.0, value=0.0, step=.001, format='%0.3f')
        
        time_low = cols[1].number_input('Known time to failure:',
                                        min_value=0.0, value=0.0, step=.001, format='%0.3f')
        C = cols[1].number_input('The creep constant',
                                        min_value=0.0, value=20.0, step=.001, format='%0.3f',
                                        help='Usually assumed to be 20.0 - 22.0 for metals')
        
        if temp_unit == 'Celsius':
            temp_low = (temp_low * 9/5) + 32
            temp_high = (temp_high * 9/5) + 32
        
        if st.button("Show results"):
            LMP = (temp_low + 459.67) * (
                C + np.log10(time_low))
            
            time_high = creep_failure_time(temp_low, temp_high, time_low, C, print_results=False)

            st.write(f"""
                     **Results from creep failure time:**
                     """)
            
            results_creep = {
                "Description": [
                    "Temperature at which the time of failure is known",
                    "Temperature at which the time of failure is unknown",
                    "Known time to failure",
                    f"Time to failure at a temperature of {temp_high} °F",
                    "Larson-Miller parameter"],
                
                "Value": [
                    f"{temp_low} °F",
                    f"{temp_high} °F",
                    f"{time_low}",
                    f"{time_high}",
                    f"{LMP}"
                ]}
            
            df_results = pd.DataFrame(results_creep)
            st.dataframe(df_results.set_index("Description"), use_container_width=True)
                        
    if method == 'Creep rupture plot':
        with st.expander('Short Guide'):
            st.write("""
            In simple terms, this module plots the creep rupture slopes for a given 
            set of creep data. The time to failure for a given temperature can be found 
            by specifying stress and temperature values.
            """)

            st.write("""
            For the creep rupture plot, the stress and time to failure of each temperature 
            are fitted into a slope using least squares polynomial fit.
            """)

            st.latex(r"""
                    S = m \cdot \log_{10}(TTF) + c
                    """)

        with st.expander('Data format'):
            st.info("""
            Upload an excel file that contains the following columns:
            * Temp - the temperatures;
            * Stress - the stresses;
            * TTF - the time to failure at the given temperatures and stresses.
            """)
            df_show = {
                'Temp': [900,900,900,900,1000,1000,1000,1000,1000,1000,1000,1000,1100,1100,1100,1100,1100,1200,1200,1200,1200,1350,1350,1350],
                'Stress': [90,82,78,70,80,75,68,60,56,49,43,38,60.5,50,40,29,22,40,30,25,20,20,15,10],
                'TTF': [37,975,3581,9878,7,17,213,1493,2491,5108,7390,10447,18,167,615,2220,6637,19,102,125,331,3.7,8.9,31.8],
            }
            df_show = pd.DataFrame.from_dict(df_show)
            df_show.to_excel('example_creep.xlsx', index=False)
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

            temp_array = df.iloc[:, 0].values
            stress_array = df.iloc[:, 1].values
            TTF_array = df.iloc[:, 2].values

            if st.checkbox("Do you want to trace a stress and temperature values on the plot?"):
                cols = st.columns([1,1])
                stress_trace = cols[0].number_input('The stress value used to determine the time to failure',
                                                min_value=0.0, value=0.0, step=.001, format='%0.3f')
                temp_trace = cols[1].number_input('The temperature value used to determine the time to failure',
                                                min_value=0.0, value=0.0, step=.001, format='%0.3f')
            else: 
                stress_trace = temp_trace = None

            if st.button("Show plot"):
                creep_rupture_curves(temp_array, stress_array, TTF_array, stress_trace, temp_trace)