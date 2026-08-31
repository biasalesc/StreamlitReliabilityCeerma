import streamlit as st
import numpy as np
import pandas as pd
from reliability.Fitters import Fit_Everything

import distributions
import functions


def show():
    st.write("""
    In this module, you can provide your data (complete or incomplete) and fit
    either the most common parametric probability distributions in reliability
    or some non-parametric models.
    """)

    with st.expander("Data format"):
        st.info("""
        Upload an excel file that contains the following columns:
        * Time - failure or right-censored time;
        * Type - 'F' for failure time, 'C' for right-censored time.
        """)
        df = {'Time': [10, 15, 8, 20, 21, 12, 13, 30, 5],
            'Type': ['F', 'F', 'C', 'F', 'C', 'C', 'F', 'F', 'C']}
        df = pd.DataFrame.from_dict(df)
        st.write(df)

    header = st.checkbox("Does your data contain header?", value=True)
    head = 0 if header else None

    col2_1, col2_2 = st.columns(2)
    uploaded_file = col2_1.file_uploader("Upload a XLSX file:",
                                         type="xlsx",
                                         accept_multiple_files=False,
                                         label_visibility="collapsed")
    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=head)

        string_cols = list(df.select_dtypes(include=['object', 'string']).columns)
        first_string_col_index = df.columns.get_loc(string_cols[0])
        if first_string_col_index > 1:
            delete_cols = list(range(first_string_col_index-1))
            df = df.drop(df.columns[delete_cols], axis=1)

        col2_2.dataframe(df, width="stretch")
        df.iloc[:,1] = df.iloc[:,1].str.upper()
        fdata = df[df.iloc[:,1] == 'F']
        cdata = df[df.iloc[:,1] == 'C']
        fdata = np.array(fdata.iloc[:,0])
        cdata = np.array(cdata.iloc[:,0])

    mod = st.selectbox("Which type of models would you like to fit?",
                       ("Parametric", "Non-parametric"))

    if mod == "Parametric":

        distr = distributions.distributions
        fitdistr = distributions.fit_distributions

        include = st.multiselect('Choose which distribution(s) you want to \
                                 fit your data to:', list(distr))

        metric, method = functions.fit_options()

        plot_params = functions.plot_parameters(CI=True, IC_plot=['CDF', 'SF'])

        exclude = list(set(list(distr))-set(include))

        exc = []
        for dis in exclude:
            for fitdis in fitdistr:
                if fitdis[:4] in dis:
                    exc.append(fitdis)

        # Weibull_Mixture / Weibull_CR / Weibull_DS are always fitted by
        # Fit_Everything by default but aren't supported by fit_distributions
        # (see distributions.unsupported_distributions) - exclude them
        # unconditionally so they can never be selected as "best" and crash
        # the lookup below with a KeyError.
        exc.extend(distributions.unsupported_distributions)

    elif mod == "Non-parametric":

        distr = distributions.non_parametric_distributions

        col1, col2 = st.columns([1,1])

        include = col1.selectbox('Choose which distribution you want to \
                                 fit your data to:', list(distr))

        ci = col2.number_input('Confidence Interval:',
                               min_value=0.0, max_value=1.0, value=0.95)

        par = st.checkbox('Do you want to compare the results with \
                          a parametric distribution?')

        if par == True:
            metric, method = functions.fit_options()

        plot_params = functions.plot_parameters(n_points=False, stats=False,
                                                IC_plot=['CDF', 'SF', 'CHF'],
                                                np=True)

    if st.button("Fit distribution"):

        if not uploaded_file:
            st.error('Please upload a file first!')
            st.stop()

        if mod == 'Parametric':

            if not include:
                st.error('Please, choose at least one distribution to fit.')
                st.stop()

            exc = exc if exc else None

            results = Fit_Everything(failures=fdata,
                                     right_censored=cdata,
                                     exclude=exc,
                                     sort_by=metric, method=method,
                                     print_results=False,
                                     show_histogram_plot=False,
                                     show_PP_plot=False,
                                     show_probability_plot=False)

            st.write('### Results of all fitted distributions')

            # Fit_Everything's own sort_by='Log-likelihood' sorts by the
            # *absolute value* of the log-likelihood in ascending order,
            # which is wrong: a higher (less negative) log-likelihood means
            # a better fit, not one closer to zero. That bug also drives
            # best_distribution/best_distribution_name, so instead of
            # trusting those we re-sort results.results ourselves with the
            # correct direction for whichever metric was chosen and pick the
            # top row as the best fit.
            sort_columns = {
                'BIC': ('BIC', True),
                'AICC': ('AICc', True),
                'AIC': ('AICc', True),
                'AD': ('AD', True),
                'LOG-LIKELIHOOD': ('Log-likelihood', False),
            }
            sort_col, ascending = sort_columns.get(metric.upper(), ('BIC', True))
            sort_key = pd.to_numeric(results.results[sort_col], errors='coerce')
            results_sorted = (
                results.results.assign(_sort_key=sort_key)
                .sort_values(by='_sort_key', ascending=ascending, na_position='last')
                .drop(columns='_sort_key')
            )

            results_table = results_sorted.set_index('Distribution')
            numeric_cols = results_table.select_dtypes(include='number').columns
            column_config = {
                col: st.column_config.NumberColumn(format=f'%.{plot_params["decimals"]}f')
                for col in numeric_cols
            }
            st.dataframe(results_table, width="stretch", column_config=column_config)

            distribution_name = results_sorted['Distribution'].values[0]
            st.write(f'### Results of the best fitted distribution: {distribution_name}')

            percentiles = np.linspace(1, 99, num=99)

            new_fit = fitdistr[distribution_name][0](
                failures=fdata, right_censored=cdata,
                show_probability_plot=False, print_results=False,
                percentiles=percentiles, method=method,
                CI=plot_params['confidence_interval']
            )
            # Use the distribution fitted here (consistent with the CI table
            # below) rather than results.best_distribution, which may still
            # reflect Fit_Everything's own (buggy) selection.
            dist = new_fit.distribution

            if 'Exponential' in distribution_name:
                new_fit.results = new_fit.results.drop(1, axis=0)

            new_fit.results = new_fit.results.set_index('Parameter')

            params_upper = list(new_fit.results.iloc[:,3])
            params_lower = list(new_fit.results.iloc[:,2])
            fit_up = fitdistr[distribution_name][1]['distribution'](
                *params_upper
            )
            fit_lw = fitdistr[distribution_name][1]['distribution'](
                *params_lower
            )

            functions.plot_distribution(dist, plot_params,
                                        title='Evaluated data',
                                        sidetable=new_fit.results,
                                        plot_dists_up_lw=plot_params['up_lw'],
                                        dist_upper=fit_up,
                                        dist_lower=fit_lw,
                                        failure_data=fdata,
                                        censored_data=cdata)

        elif mod == "Non-parametric":

            dist = distr[include](failures=fdata, right_censored=cdata, CI=ci,
                                  show_plot=False, print_results=False)

            # Parametric comparison
            if par == True:
                results = Fit_Everything(failures=fdata,
                                         right_censored=cdata,
                                         sort_by=metric, method=method,
                                         print_results=False,
                                         show_histogram_plot=False,
                                         show_PP_plot=False,
                                         show_probability_plot=False)
                par_params = [results]
            else:
                par_params = None

            functions.plot_distribution(dist, plot_params,
                                        title='Evaluated data',
                                        plot_dists=['CDF', 'SF', 'CHF'],
                                        plot_dists_up_lw=plot_params['up_lw'],
                                        non_parametric=True, par=par_params,
                                        failure_data=fdata,
                                        censored_data=cdata)
