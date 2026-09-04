import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from reliability.Distributions import Normal_Distribution, Beta_Distribution
from reliability.Other_functions import stress_strength, stress_strength_normal

import distributions
import functions


def show():
    st.write("""
    In this module, you can predict the probability of failure when both
    Stress and Strength probability distributions are known.
    """)

    with st.expander('Short Guide'):
        st.write("""
        In simple terms, a failure is defined as when the stress to which an
        item is subjected to exceeds the corresponding strength.

        If both the Stress and the Strength distributions are Normal
        distributions, there is an analytical solution for the probability
        of failure:
        """)
        st.latex(r"""
        \text{Probability of failure} =\Phi \left(\frac{\mu_{strength}-
        \mu_{stress}}{\sqrt{\sigma_{strength}^{2}+\sigma_{stress}^{2}}}\right)
        """)

        st.write("""
        If any of both are not Normal distributions, then
        the integration method is required:
        """)
        st.latex(r"""
        \text{Probability of failure} =\int^{\infty}_{0}
        \left( f_{strength} \times R_{stress} \right)
        """)

    distr = distributions.distributions

    models = ['Stress', 'Strength']

    selected_dists = []

    for i in models:
        cols = st.columns([3,1,1,1])
        distribution_name = cols[0].selectbox(f'Select distribution \
                                              to model the {i}:',
                                              list(distr),
                                              key=f"dist_{i}")

        var = []
        variables = distr[distribution_name]['variables']
        for j, variable in enumerate(variables):
            item = variables[variable]
            to_append = cols[j+1].number_input(variable[-1],
                                               value=item[0],
                                               min_value=item[1],
                                               step=0.1,
                                               format='%0.2f',
                                               key=f'var_{j}_{i}')
            var.append(to_append)

        selected_dists.append(distr[distribution_name]['distribution'](*var))

    stress, strength = selected_dists

    if st.button('Calculate'):
        # If both are normal, calculate prob of failure by exact method
        if (type(stress) == Normal_Distribution) and \
           (type(strength) == Normal_Distribution):

            prob_failure = stress_strength_normal(stress=stress,
                                                  strength=strength,
                                                  show_plot=False,
                                                  print_results=False)

        # If not, use numerical integration method
        else:
            prob_failure = stress_strength(stress=stress,
                                           strength=strength,
                                           show_plot=False,
                                           print_results=False)

        xmin = stress.quantile(0.00001)
        xmax = strength.quantile(0.99999)
        if abs(xmin) < (xmax - xmin) / 4:
            xmin = 0
        if type(stress) == Beta_Distribution:
            xmin = 0
        if type(strength) == Beta_Distribution:
            xmax = 1
        xvals = np.linspace(xmin, xmax, 10000)
        stress_PDF = stress.PDF(xvals=xvals, show_plot=False)
        strength_PDF = strength.PDF(xvals=xvals, show_plot=False)

        st.write('Stress Distribution:', stress.param_title_long)
        st.write('Strength Distribution:', strength.param_title_long)
        st.write(f'Probability of failure (stress > strength): \
                 **{prob_failure*100:.5f}%**')

        colors = px.colors.qualitative.Set1

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xvals, y=stress_PDF, mode='lines',
                                 name='Stress', marker=dict(color=colors[0]),
                                 visible=True, fill='tozeroy'))
        fig.add_trace(go.Scatter(x=xvals, y=strength_PDF, mode='lines',
                                 name='Strength', marker=dict(color=colors[1]),
                                 visible=True, fill='tozeroy'))
        update_menus = functions.plot_parameters(log_only=True)
        fig = functions.update_layout(fig, title='Stress and Strength model',
                                      xtitle='Stress/Strength units',
                                      tick_format='0.2f',
                                      update_menus=update_menus)
        st.plotly_chart(fig, width="stretch")
    else:
        if stress.mean > strength.mean:
            st.error("Warning: strength mean must be greater \
                     than or equal to stress mean")
