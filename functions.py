import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reliability.Probability_plotting import plotting_positions
from reliability.Utils import round_and_string


def round_to_decimals(value, decimals):
    return float(round_and_string(value, decimals))


def write_text_align(text='Hello World!', align='justify'):
    assert align in {'center', 'left', 'right', 'justify'},\
        'Text align must be one of: "center", "left", "right" or "justify"'
    st.markdown(
        f'<div style="text-align: {align};">{text}</div>',
        unsafe_allow_html=True
    )


def page_config(title=None, hide_menu=False, reduce_whitespace=False):
    if hide_menu:
        element = """
        <style>
        #MainMenu {visibility: hidden;}
        #ReportStatus {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
    else:
        element = """
        <style>
        footer {visibility: hidden;}
        </style>
        """
    if title:
        element = element + f"""
        <div style="display:table;\
            font-size:48px;font-weight:bold;margin-top:-50px">
        {title}
        </div>
        """
    if reduce_whitespace:
        pass
        # element = element + """
        # <style>
        #        .block-container {
        #             padding-top: 0rem;
        #             padding-bottom: 0rem;
        #             padding-left: 3rem;
        #             padding-right: 3rem;
        #         }
        # </style>
        # """

    st.markdown(element, unsafe_allow_html=True)
    st.markdown("""
    <style>
    a[href*="streamlit.io"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


def plot_parameters(*, n_points=True, decimals=True, stats=True,
                    log_only=False, CI=False, IC_plot=None, np=False):
    output = {}

    output['display_log_options'] = [
        dict(
            buttons=list([
                dict(
                    args=[{'xaxis.type': '-', 'yaxis.type': '-'}],
                    label='Linear',
                    method='relayout'
                ),
                dict(
                    args=[{'xaxis.type': 'log', 'yaxis.type': '-'}],
                    label='Log-x',
                    method='relayout'
                ),
                dict(
                    args=[{'xaxis.type': '-', 'yaxis.type': 'log'}],
                    label='Log-y',
                    method='relayout'
                ),
                dict(
                    args=[{'xaxis.type': 'log', 'yaxis.type': 'log'}],
                    label='Log-xy',
                    method='relayout'
                ),
            ]),
            direction="right",
            type="buttons",
            pad={"r": 10, "t": 10},
            x=0.0,
            xanchor="left",
            y=1.115,
            yanchor="top"
        )
    ]

    if log_only:
        return output['display_log_options']

    exp = st.expander("Plot parameters")

    if n_points:
        output['plot_points'] = exp.number_input('Number of points to plot:',
                                                 min_value=5,
                                                 value=5000,
                                                 max_value=100000)
        output['plot_points'] = int(output['plot_points'])

    if decimals:
        output['decimals'] = exp.number_input('Number of decimal places:',
                                              min_value=0,
                                              value=2,
                                              max_value=None)
        output['decimals'] = int(output['decimals'])

    if CI:
        output['confidence_interval'] = exp.number_input(
            'Confidence Interval:',
            min_value=0.0, max_value=1.0, value=0.80
            )

    if stats:
        output['plot_mean'] = exp.checkbox("Show distribution mean.",
                                           value=True)
        output['plot_median'] = exp.checkbox("Show distribution median.",
                                             value=True)
        output['plot_mode'] = exp.checkbox("Show distribution mode.",
                                           value=True)
    else:
        output['plot_mean'] = False
        output['plot_median'] = False
        output['plot_mode'] = False

    output['plot_e10'] = exp.checkbox("Show axis values in 0.0e+0 form.",
                                      value=False)

    if IC_plot:
        if np:
            funcs = ['CDF', 'SF', 'CHF']
        else:
            funcs = ['PDF', 'CDF', 'SF', 'HF', 'CHF']
        output['up_lw'] = exp.multiselect(
            "Choose which functions to show upper and lower IC:",
            funcs, IC_plot
            )

    return output


def fit_options(alt=False, include_CI=False):
    if include_CI:
        cols = st.columns([1,1,1])
    else:
        cols = st.columns([1,1])

    if alt:
        metric = cols[0].radio('Choose a goodness of fit criteria:',
                            ('BIC', 'AICc', 'Log-likelihood'))
        method = cols[1].radio('Choose the optimizer:', ('TNC', 'L-BFGS-B'))
    else:
        metric = cols[0].radio('Choose a goodness of fit criteria:',
                            ('BIC', 'AICc', 'AD', 'Log-likelihood'))
        method = cols[1].radio('Choose the method to fit the distribution:',
                            ('MLE', 'LS', 'RRX', 'RRY'))

    if include_CI:
        ci = cols[2].number_input(
            'Confidence Interval:',
            min_value=0.0, max_value=1.0, value=0.80
            )
        return metric, method, ci

    return metric, method


def plot_distribution(dist, plot_params, *, title='',
                      sidetable=None, plot=True,
                      plot_dists=['PDF', 'CDF', 'SF', 'HF', 'CHF'],
                      dist_upper=None, dist_lower=None,
                      plot_dists_up_lw=['SF'],
                      failure_data=None, censored_data=None,
                      non_parametric=False, par=None):

    if not non_parametric:
        properties = {
            'Mean': \
                f"{dist.mean:.{plot_params['decimals']}f}",
            'Median': \
                f"{dist.median:.{plot_params['decimals']}f}",
            'Mode': \
                f"{dist.mode}" if isinstance(dist.mode, str) \
                else f"{dist.mode:.{plot_params['decimals']}f}",
            'Variance': \
                f"{dist.variance:.{plot_params['decimals']}f}",
            'Std. Deviation': \
                f"{dist.standard_deviation:.{plot_params['decimals']}f}",
            'Skewness': \
                f"{dist.skewness:.{plot_params['decimals']}f}",
            'Kurtosis': \
                f"{dist.kurtosis:.{plot_params['decimals']}f}",
            'Excess Kurtosis': \
                f"{dist.excess_kurtosis:.{plot_params['decimals']}f}",
        }

        properties_text = fr""
        for statistic in properties:
            properties_text += f"{statistic}: {properties[statistic]}  \n"

        properties_table = pd.DataFrame.from_dict(
            properties,
            orient='index',
            columns=['Statistic'],
        )

        if sidetable is not None:
            cols = st.columns([2,1])
            cols[0].dataframe(sidetable, width="stretch")
            # cols[1].write(properties_text)
            cols[1].dataframe(properties_table, width="stretch")
        else:
            cols = st.columns([1])
            # cols[0].write(properties_text)
            cols[0].dataframe(properties_table, width="stretch")

    if plot:
        # Points of X axis
        if non_parametric:
            x = dist.xvals
        else:
            dist.PDF()
            x_min, x_max = plt.gca().get_xlim()
            x = np.linspace(float(x_min), float(x_max),
                            int(plot_params['plot_points']))

        # Create empty figure
        fig = go.Figure()

        # Get colors for plot
        colors = px.colors.qualitative.Set1
        color = {
            'PDF': colors[0],
            'CDF': colors[1],
            'SF': colors[2],
            'HF': colors[3],
            'CHF': colors[4],
        }

        # Add each distribution to the figure
        for distrib_str in plot_dists:
            distrib_func = getattr(dist, distrib_str)
            if non_parametric:
                y_distrib = distrib_func
                visible = True
            else:
                y_distrib = distrib_func(xvals=x)
                if distrib_str == 'PDF':
                    visible = True
                else:
                    visible = 'legendonly'

            fig.add_trace(go.Scatter(x=x, y=y_distrib,
                                     mode='lines', name=distrib_str,
                                     marker=dict(color=color[distrib_str]),
                                     opacity=0.9, visible=visible))

        condition1 = (dist_upper is not None) and \
            (dist_lower is not None) and (not non_parametric)

        # Add upper and lower plots if given or if non-parametric
        if condition1 or non_parametric:
            for distrib_str in plot_dists_up_lw:
                if non_parametric:
                    y_up = getattr(dist, distrib_str+'_upper')
                    y_lw = getattr(dist, distrib_str+'_lower')
                else:
                    y_up = getattr(dist_upper, distrib_str)(xvals=x)
                    y_lw = getattr(dist_lower, distrib_str)(xvals=x)

                fig.add_trace(go.Scatter(x=x, y=y_lw, mode='lines',
                                         name=distrib_str + ' lower',
                                         marker=dict(color=color[distrib_str]),
                                         opacity=0.3, visible='legendonly'))
                fig.add_trace(go.Scatter(x=x, y=y_up, mode='lines',
                                         name=distrib_str + ' upper',
                                         marker=dict(color=color[distrib_str]),
                                         opacity=0.3, visible='legendonly',
                                         fill='tonexty',
                                         fillcolor='rgba' + \
                                            color[distrib_str][3:-5]+',0.3)'))

        # Add failure and censored plots if given
        if (failure_data is not None):
            fig.add_trace(go.Histogram(x=failure_data,
                                       histnorm='probability density',
                                       name='Original data (failure)',
                                       marker=dict(color=color['PDF']),
                                       opacity=0.5, visible=True))
            if (censored_data is not None):
                x_f, y_f = plotting_positions(failures=failure_data,
                                              right_censored=censored_data)
                fig.add_trace(go.Scatter(x=x_f, y=y_f, mode='markers',
                                         name='Failure points',
                                         marker=dict(color=color['CDF']),
                                         visible='legendonly'))

        # If non-parametric but wants to compare with a parametric
        if non_parametric and par:
            results = par[0]
            for distrib_str in plot_dists:
                distrib_func = getattr(results.best_distribution, distrib_str)
                y_distrib = distrib_func(xvals=x)
                name = distrib_str+\
                    f' parametric: {results.best_distribution_name}'
                fig.add_trace(go.Scatter(x=x, y=y_distrib, mode='lines',
                                         name=name,
                                         marker=dict(color=color[distrib_str]),
                                         visible=True, line=dict(dash='dot')))

        # Add statistics if desired
        if plot_params['plot_mean']:
            fig.add_vline(x=dist.mean, line_dash="dash",
                          annotation_text="Mean",
                          annotation_position="top right")
        if plot_params['plot_median']:
            fig.add_vline(x=dist.median, line_dash="dash",
                          annotation_text="Median",
                          annotation_position="top right")
        if plot_params['plot_mode']:
            fig.add_vline(x=dist.mode, line_dash="dash",
                          annotation_text="Mode",
                          annotation_position="top right")
        if plot_params['plot_e10']:
            tick_format = f"0.{plot_params['decimals']}e"
        else:
            tick_format = f"0.{plot_params['decimals']}f"

        # Plot layout
        fig = update_layout(fig, title=title, xtitle='Time',
                            tick_format=tick_format,
                            update_menus=plot_params['display_log_options'])

        st.plotly_chart(fig, width="stretch")


def update_layout(fig, *, title='', xtitle='',
                  ytitle='Probability/Density', tick_format, update_menus):
    fig.update_layout(title_text=title,
                    #   width=1900, height=600,
                      yaxis=dict(tickformat=tick_format),
                      xaxis=dict(tickformat=tick_format),
                      updatemenus=update_menus,
                      font_family="Times New Roman",
                      title_font_family="Times New Roman",
                      legend={'traceorder': 'normal'})
    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title=ytitle)

    return fig
