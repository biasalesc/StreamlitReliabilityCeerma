import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO, BytesIO
import contextlib
import functions
import distributions
from lifelines.plotting import plot_lifetimes
from lifelines import (
    KaplanMeierFitter,
    CoxPHFitter,
    WeibullAFTFitter,
    LogNormalAFTFitter,
    LogLogisticAFTFitter
)
from lifelines.utils import ConvergenceError
import matplotlib.pyplot as plt
import lifelines.plotting as lifelines_plotting
import traceback

LIFELINES_MODELS = {
    "Kaplan-Meier": KaplanMeierFitter,
    "CoxPH": CoxPHFitter,
    "Weibull AFT": WeibullAFTFitter,
    "LogNormal AFT": LogNormalAFTFitter,
    "LogLogistic AFT": LogLogisticAFTFitter,
}

def create_right_censored_excel_example():
    """Cria um arquivo Excel de exemplo para dados censurados à direita."""
    data = {
        'Time': [620, 632, 685, 822, 380, 416, 460, 596, 216, 146, 332, 400],
        'Type': ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        'Stress1': [348, 348, 348, 348, 348, 348, 348, 348, 378, 378, 378, 378],
        'Stress2': [3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3],
    }
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='ALT_Data')
    return output.getvalue()

def create_left_censored_excel_example():
    """Cria um arquivo Excel de exemplo para dados censurados à esquerda."""
    data = {
        'Time': [620, 632, 50, 822, 380, 100, 460, 596, 30, 146, 332, 400],
        'Type': ['F', 'F', 'L', 'C', 'F', 'L', 'F', 'C', 'L', 'F', 'F', 'F'],
        'Stress1': [348, 348, 348, 348, 348, 348, 348, 348, 378, 378, 378, 378],
        'Stress2': [3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3],
    }
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='ALT_Data')
    return output.getvalue()

def create_interval_censored_excel_example():
    """Cria um arquivo Excel de exemplo para dados com censura por intervalo."""
    data = {
        'Time_Lower': [100, 200, pd.NA, 50, 150, 0, 500],
        'Time_Upper': [100, pd.NA, 50, 70, np.inf, 20, 500],
        'Event': [1, 0, 1, 1, 0, 1, 1],
        'Stress1': [348, 348, 348, 348, 350, 350, 355],
        'Stress2': [3, 3, 3, 3, 5, 5, 3],
    }
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='ALT_Data')
    return output.getvalue()

# --- Funções Auxiliares ---
def detect_censoring_from_headers(df_columns):
    """
    Detects the type of censoring based on a list of DataFrame column headers.

    Args:
        df_columns (list): A list of column header strings from the DataFrame.

    Returns:
        str or None: A string indicating the detected censoring type
                     ("interval", "left", "right", "unknown_time_type_format", "incomplete_interval_format", "ambiguous")
                     or None if headers are insufficient.
    """
    if not df_columns:
        return None

    # Normalize column names: lowercase and replace spaces/common separators with underscores
    normalized_cols = [str(col).strip().lower().replace(" ", "_").replace("-", "_") for col in df_columns]

    # --- Check for Interval Censoring (highest priority due to distinct column names) ---
    has_time_lower = any(col in ['time_lower', 'lower_bound', 't_low'] for col in normalized_cols)
    has_time_upper = any(col in ['time_upper', 'upper_bound', 't_upp'] for col in normalized_cols)
    has_event_col = 'event' in normalized_cols # 'event' is quite specific for interval in this context

    if has_time_lower and has_time_upper and has_event_col:
        # Could add checks for Stress1, Stress2 if they are strictly required for interval
        # For now, these three are the defining ones for interval format.
        return "interval"
    elif has_time_lower or has_time_upper or has_event_col:
        # If some but not all interval-specific columns are present, it's an incomplete format
        return "incomplete_interval_format" # Or "ambiguous" if it might be something else

    # --- Check for Time/Type based Censoring (Left/Right) ---
    has_time_col = 'time' in normalized_cols
    has_type_col = 'type' in normalized_cols

    if has_time_col and has_type_col:
        # At this point, we assume it's either 'left' or 'right' based on content.
        # The function signature asks to return based on *headers*.
        # So, if Time and Type are present, we'd need to inspect df content for 'L'.
        # For header-only detection, we can only say it's "time_type_format".
        # The main application logic will then inspect the 'Type' column's content.
        return "time_type_format" # Indicates that 'Type' column content needs checking

    # --- Handle ambiguous or insufficient headers ---
    if has_time_col and not has_type_col:
        return "ambiguous_time_no_type" # Has 'Time' but missing 'Type'
    if not has_time_col and has_type_col:
        return "ambiguous_type_no_time" # Has 'Type' but missing 'Time'

    return "unknown" # None of the recognized patterns match
def explain_convergence_error():
    # ... (sem mudanças)
    st.markdown("""
    **What does 'Did not converge' mean?**

    When a statistical model "does not converge," it means the algorithm used to estimate
    the model parameters (the numbers that define the model) could not find a stable
    solution. Imagine trying to find the lowest point in a hilly landscape blindfolded;
    if the landscape is too complex or you take wrong steps, you might wander forever
    or get stuck somewhere that isn't the true bottom.

    **Possible reasons for AFT model non-convergence:**
    * **Insufficient Data:** Not enough data points, especially distinct failure times,
        to reliably estimate the model's shape and scale.
    * **Data Sparsity:** If failures are very spread out or rare.
    * **Poor Initial Guesses:** The algorithm starts with an initial guess for parameters;
        if this guess is too far off, it might not find its way to a good solution.
    * **Flat Likelihood Surface:** The "landscape" the algorithm is searching might be
        very flat, meaning many different parameter sets give almost equally good fits,
        making it hard to pick one.
    * **Collinearity (if using stress variables):** If stress variables are highly
        correlated, their individual effects are hard to disentangle.
    * **Model Misspecification:** The chosen distribution (e.g., Weibull, LogNormal)
        might be a very poor fit for the actual underlying failure process.

    **What to do?**
    * Ensure your data is clean and correctly formatted.
    * Try a different AFT distribution if one fails.
    * If using stress variables, check for high correlations.
    * More data or data with more precise failure times often helps.
    """)

def show():
    st.title("ALT model estimation")
    st.write("""
    In this module, you can provide your Accelerated Life Testing (ALT)
    data (complete or incomplete) and fit the most common probability
    distributions in reliability.
    """)
    st.markdown("""
    This tool helps you analyze Accelerated Life Testing (ALT) data with various types of censoring.
    Please prepare your data in an Excel (.xlsx) file according to one of the formats below.
    The system will attempt to auto-detect the data type upon upload.
    """)

    with st.expander('Complete data and right Censored Data'):
        st.markdown("""
        **Use Case:** You know the exact time of failure, or you know that a unit was still working at a certain time (right-censored).
        **Required Columns (case-insensitive, spaces/underscores flexible):**
        * `Time`: The failure time or the time of right-censoring.
        * `Type`:
            * `'F'` for an observed Failure.
            * `'C'` for a Right-Censored observation.
        * `Stress1`: The level of the primary stress factor.
        * `Stress2` (Optional): The level of a second stress factor.
        **Analysis Library:** Custom `reliability` library.
        """)
        st.write("Example Excel Structure for Complete and Left Censored Data:")
        df_show = {
            'Time': [620,632,685,822,380,416,460,596,216,146,332,400],
            'Type': ['F','F','F','F','F','F','F','F','F','F','F','F'],
            'Stress1':[348,348,348,348,348,348,348,348,378,378,378,378],
            'Stress2': [3,3,3,3,5,5,5,5,3,3,3,3],
        }
        df_show = pd.DataFrame.from_dict(df_show)
        st.write(df_show)
        st.download_button(
            label="Download Example Excel File",
            data=create_right_censored_excel_example(),
            file_name="alt_right_censored_example.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.info('The use level stress parameter is optional. \
                If single stress model, enter only one value. For example:')
        st.write('323')
        st.info('If dual stress model, enter two values \
                separated by ";". For example:')
        st.write('323; 2')

    with st.expander('Complete data, right and left Censored Data'):
        st.markdown("""
        **Use Case:** You know that a failure occurred *before* a certain inspection time, but not exactly when. Also handles exact failures and right-censored data.
        **Required Columns (case-insensitive, spaces/underscores flexible):**
        * `Time`: The inspection time (for left/right censored) or exact failure time.
        * `Type`:
            * `'F'` for an observed Failure.
            * `'C'` for a Right-Censored observation.
            * `'L'` for a Left-Censored observation (event occurred *before* `Time`).
        * `Stress1`: The level of the primary stress factor.
        * `Stress2` (Optional): The level of a second stress factor.
        **Analysis Library:** `lifelines`.
        """)
        st.write("Example Excel Structure for Left Censored Data:")
        df_show_dict = {
            'Time': [620, 632, 50, 822, 380, 100, 460, 596, 30, 146, 332, 400],
            'Type': ['F', 'F', 'L', 'C', 'F', 'L', 'F', 'C', 'L', 'F', 'F', 'F'],
            'Stress1': [348, 348, 348, 348, 348, 348, 348, 348, 378, 378, 378, 378],
            'Stress2': [3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3],
        }
        df_show = pd.DataFrame(df_show_dict)
        st.write(df_show)
        st.download_button(
            label="Download Example Excel File",
            data=create_left_censored_excel_example(),
            file_name="alt_left_censored_example.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.info('The use level stress parameter is optional. \
                If single stress model, enter only one value (e.g., `323`). \
                If dual stress model, enter two values separated by ";" (e.g., `323; 2`).')

    with st.expander("Complete data, right, left and interval Censored Data"):
        st.markdown("""
        **Use Case:** You know that a failure occurred within a specific time interval $[T_{lower}, T_{upper}]$. This format can also represent exact, left, and right censored data.
        **Required Columns (case-insensitive, spaces/underscores flexible):**
        * `Time_Lower` (or `Lower_Bound`): Lower bound of the interval. For left-censored, use 0 or leave blank (will be treated as 0). For right-censored, this is the last observation time.
        * `Time_Upper` (or `Upper_Bound`): Upper bound of the interval. For right-censored, use `inf` or leave blank. For left-censored, this is the inspection time. For exact, `Time_Upper` = `Time_Lower`.
        * `Event`:
            * `1`: If the event is observed within the interval (exact, left, or true interval).
            * `0`: If the observation is right-censored (event NOT observed by `Time_Lower`).
        * `Stress1`: The level of the primary stress factor.
        * `Stress2` (Optional): The level of a second stress factor.
        **Analysis Library:** `lifelines`.
        """)
        st.write("Interpreting `Time_Lower`, `Time_Upper`, `Event` for Interval Censoring:")
        st.info("""
        - **Exact Failure at T:** `Time_Lower`=T, `Time_Upper`=T, `Event`=1
        - **Left-Censored (failure before T_upper):** `Time_Lower`=0 (or blank), `Time_Upper`=T_upper, `Event`=1
        - **Right-Censored (survival beyond T_lower):** `Time_Lower`=T_last_obs, `Time_Upper`=inf (or blank), `Event`=0
        - **Interval-Censored (failure between T_lower, T_upper):** `Time_Lower` < `Time_Upper`, `Event`=1
        """)
        st.write("Example Excel Structure for interval Censored Data:")
        df_show_dict_interval = {
            'Time_Lower': [100,    200,      pd.NA, 50,  150,   0,      500],
            'Time_Upper': [100,    pd.NA,    50,    70,  np.inf, 20,     500],
            'Event':      [1,      0,        1,     1,   0,      1,      1],
            'Stress1':    [348,    348,      348,   348, 350,   350,    355],
            'Stress2':    [3,      3,        3,     3,   5,     5,      3],
        }
        df_show_interval = pd.DataFrame(df_show_dict_interval)
        st.write("Example DataFrame Structure (User Input):")
        st.dataframe(df_show_interval)
        st.download_button(
            label="Download Example Excel File",
            data=create_interval_censored_excel_example(),
            file_name="alt_interval_censored_example.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.info("""
        - Blank/NaN cells in `Time_Lower` will be treated as 0.
        - Blank/NaN cells in `Time_Upper` (when `Event=0`) will be treated as `numpy.inf`.
        - If `Event=1` (left/interval/exact), `Time_Upper` must be a finite number (or equal to `Time_Lower` for exact). Blank `Time_Upper` with `Event=1` is an error.
        - Ensure `Time_Lower <= Time_Upper`.
        """)

    col2_1, col2_2 = st.columns(2)
    uploaded_file = col2_1.file_uploader("Upload a XLSX file",
                                         type="xlsx",
                                         accept_multiple_files=False,
                                         label_visibility="collapsed")

    if 'detected_censoring_type' not in st.session_state:
        st.session_state.detected_censoring_type = None
    if 'df_columns_for_detection' not in st.session_state:
        st.session_state.df_columns_for_detection = None
    st.session_state.final_censoring_type = None
    if uploaded_file is not None:
        try:
            df_header_only = pd.read_excel(uploaded_file, nrows=0)
            df_columns = list(df_header_only.columns)
            st.session_state.df_columns_for_detection = df_columns


            header_based_detection = detect_censoring_from_headers(df_columns)
            st.session_state.detected_censoring_type = header_based_detection

            if header_based_detection == "time_type_format":
                df_full = pd.read_excel(uploaded_file, header=0 if st.checkbox("File has header row", True) else None)
                type_col_name = None
                for col in df_full.columns:
                    if str(col).strip().lower().replace(" ", "_") == "type":
                        type_col_name = col
                        break

                if type_col_name and type_col_name in df_full.columns:
                    if df_full[type_col_name].astype(str).str.upper().str.contains('L').any():
                        st.session_state.final_censoring_type = "left"
                        st.success("Final Detected Censoring Type: **Left Censored** ")
                    else:
                        st.session_state.final_censoring_type = "right"
                        st.success("Final Detected Censoring Type: **Right Censored or No cesored data**")
                else:
                    st.session_state.final_censoring_type = "unknown_time_type_format_error"
                    st.error("Error: 'Time/Type' format indicated, but 'Type' column could not be reliably processed.")

            elif header_based_detection == "interval":
                st.session_state.final_censoring_type = "interval"
                st.success("Final Detected Censoring Type: **Interval Censored**")
            elif header_based_detection == "incomplete_interval_format":
                st.session_state.final_censoring_type = "incomplete_interval_format"
                st.warning("Warning: Headers suggest an **Incomplete Interval Censoring Format**. Please check your columns (`Time_Lower`, `Time_Upper`, `Event`).")
            elif header_based_detection in ["ambiguous_time_no_type", "ambiguous_type_no_time", "unknown"]:
                st.session_state.final_censoring_type = header_based_detection
                st.error(f"Error: Could not reliably determine censoring type from headers. Status: **{header_based_detection}**. Please check column names.")
            else:
                st.session_state.final_censoring_type = "unknown_error"
                st.error("An unexpected error occurred during censoring type detection.")


        except Exception as e:
            st.error(f"Error reading or processing Excel file: {e}")
            st.session_state.detected_censoring_type = None
            st.session_state.df_columns_for_detection = None
            st.session_state.final_censoring_type = None
    else:
        st.info("Please upload an Excel file to detect the censoring type.")
        if st.session_state.detected_censoring_type is not None or \
           st.session_state.df_columns_for_detection is not None:
            st.session_state.detected_censoring_type = None
            st.session_state.df_columns_for_detection = None
            st.session_state.final_censoring_type = None

    if st.session_state.final_censoring_type == "right":
        with st.expander('Short Guide'):
            st.write('When using this module, please take into consideration \
                    the following points:')
            st.info("""
            - There is no need to sort the data in any particular order as this
            is all done automatically;
            - For single-stress, number of failure points must be at least three
            for each stress level;
            - For dual-stress, number of failure points must be at least four for
            each combination of stress levels.
            **Right-censored data** means the event had not occurred by the specified time.
            """)
            st.write("""
            Accelerated Life Testing is implemented on Exponential, Weibull,
            Normal and Lognormal distributions. One of the parameters of the
            distribution will be changed into a Life Model function L(S), which
            models the behaviour of the stress changes. The parameter changed
            is as follows:
            """)
            for item in distributions.alt_substitution_equations:
                st.latex(item)

            st.info("- L(S) function options are shown in Equation Information tab.")

        with st.expander("Equation Information"):
            st.write("Life Model functions available:")
            st.write("- Single-stress:")
            for item in distributions.alt_single_equations:
                st.latex(item)
            st.write("- Dual-stress:")
            for item in distributions.alt_dual_equations:
                st.latex(item)
            st.info("""Although named Power-Exponential, this Life-Stress Model
            actually applies Exponential to stress 1 and Power to stress 2. You
            can simply shift the order of your input columns to switch which
            stress is affected by each function.""")
        dual = False
        if uploaded_file:
            df = pd.read_excel(uploaded_file)

            string_cols = list(df.select_dtypes(include=['object', 'string']).columns)
            first_string_col_index = df.columns.get_loc(string_cols[0])
            if first_string_col_index > 1:
                delete_cols = list(range(first_string_col_index-1))
                df = df.drop(df.columns[delete_cols], axis=1)

            n_cols = len(df.columns)
            if n_cols <= 2 or n_cols >= 5:
                st.error('Please enter data according to the \
                        "Data format" example!')
                st.stop()
            else:
                col2_2.dataframe(df, width="stretch")
                df.iloc[:,1] = df.iloc[:,1].str.upper()
                fdata = df[df.iloc[:,1] == 'F']
                cdata = df[df.iloc[:,1] == 'C']
                ftime = np.array(fdata.iloc[:,0])
                ctime = np.array(cdata.iloc[:,0])
                fstress_1 = np.array(fdata.iloc[:,2])
                cstress_1 = np.array(cdata.iloc[:,2])
                if n_cols > 3:
                    dual = True
                    fstress_2 = np.array(fdata.iloc[:,3])
                    cstress_2 = np.array(cdata.iloc[:,3])

                n_stress1 = np.unique(df.iloc[:,2])
                if dual:
                    n_stress2 = np.unique(df.iloc[:,3])
                    for s1 in n_stress1:
                        for s2 in n_stress2:
                            aux = df[df.iloc[:,2] == s1]
                            aux = aux[aux.iloc[:,3] == s2]
                            aux = aux[aux.iloc[:,1] == 'F']
                            if len(aux) > 0 and len(aux) < 4:
                                st.error('All combinations of stress levels must have at least 4 failures!')
                                st.stop()
                else:
                    for s1 in n_stress1:
                        aux = df[df.iloc[:,2] == s1]
                        aux = aux[aux.iloc[:,1] == 'F']
                        if len(aux) > 0 and len(aux) < 4:
                            st.error('All stress levels must have at least 3 failures!')
                            st.stop()

                use_level = st.text_input("Use level stress (optional)")
                if use_level:
                    use_level = use_level.strip().split(sep=';')
                    use_level = [float(x or 0) for x in use_level]
                    if dual and len(use_level) != 2:
                        st.error('Please enter two use level stresses!')
                        st.stop()
                    elif not dual and len(use_level) != 1:
                        st.error('Please enter one use level stress only!')
                        st.stop()
                else:
                    use_level = None


        if dual:
            distr = distributions.alt_dual_distributions
        else:
            distr = distributions.alt_single_distributions

        include = st.multiselect('Choose which distribution(s) you want to \
                                fit your data to:', list(distr))

        metric, method, ic = functions.fit_options(alt=True, include_CI=True)


        if st.button("Fit ALT model"):

            if not uploaded_file:
                st.error('Please upload a file first!')
                st.stop()

            with st.spinner('Fitting models...'):
                terminal_buffer = StringIO()
                print_results = False
                probability_plot = True
                life_stress_plot = True

                if dual:
                    function_parameters = [
                        ftime, fstress_1, fstress_2, ctime, cstress_1, cstress_2,
                        use_level, ic, method,
                        probability_plot, life_stress_plot, print_results
                    ]
                else:
                    function_parameters = [
                        ftime, fstress_1, ctime, cstress_1,
                        use_level[0] if use_level else None, ic, method,
                        probability_plot, life_stress_plot, print_results
                    ]

                best_BIC, best_AICc, best_loglik = np.inf, np.inf, np.inf
                results = pd.DataFrame(
                    columns=[
                        "ALT_model",
                        "a",
                        "b",
                        "c",
                        "n",
                        "beta",
                        "sigma",
                        "Log-LH",
                        "AICc",
                        "BIC",
                    ],
                )
                model_warnings = {}
                def capture_terminal_warnings():
                    output = terminal_buffer.getvalue()
                    warnings = []
                    for line in output.split('\n'):
                        if "WARNING:" in line and ("MLE estimates failed" in line or "least squares estimates" in line):
                            clean_line = line.replace('WARNING: ', '').strip()
                            model_name = clean_line.split('for ')[1].split('.')[0].strip()
                            message = clean_line.split('. ')[1].strip()
                            warnings.append((model_name, message))
                    return warnings
                with contextlib.redirect_stderr(terminal_buffer), \
                    contextlib.redirect_stdout(terminal_buffer):

                    for item in include:
                        try:
                            res = distr[item](*function_parameters)

                        except Exception as e:
                            model_warnings[item] = f"Erro crítico: {str(e)}"
                            continue

                terminal_warnings = capture_terminal_warnings()

                for model, message in terminal_warnings:
                    model_warnings[model] = message


                for item in include:

                    res = distr[item](*function_parameters)


                    results = results.append(
                        {
                            "ALT_model": item,
                            "a": f'{res.a:.4f}' if 'a' in dir(res) else '',
                            "b": f'{res.b:.4f}' if 'b' in dir(res) else '',
                            "c": f'{res.c:.4f}' if 'c' in dir(res) else '',
                            "n": f'{res.n:.4f}' if 'n' in dir(res) else '',
                            "beta": f'{res.beta:.4f}' if 'beta' in dir(res) else '',
                            "sigma": f'{res.sigma:.4f}' if 'sigma' in dir(res) else '',
                            "Log-LH": f'{res.loglik:.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if (res.BIC < best_BIC and metric == 'BIC') or \
                    (res.AICc < best_AICc and metric == 'AICc') or \
                    (-res.loglik < best_loglik and metric == 'Log-likelihood'):
                        best_BIC = res.BIC
                        best_AICc = res.AICc
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = item

                st.write('## Results of all fitted ALT models')
                results = pd.DataFrame.from_dict(results)
                st.write(results)

                if model_warnings:
                    st.warning("**Warning - Suboptimal Results Detected**")
                    for model, message in model_warnings.items():
                        st.markdown(f"""
                        ⚠️ **Model {model.replace('_', ' ').title()}**
                        🔍 *Problem detected:*
                        {message}

                        💡 *Implications:*
                        The results were obtained with alternative methods (least squares)
                        and may be less accurate than the standard MLE method.
                        """)
                    st.write("---")

                st.write('## Results of the best fitted ALT model')
                st.write(f'The best model are: **{best_model_name}**')
                st.write(best_model.results)
                probability_plot = best_model.probability_plot.figure
                life_stress_plot = best_model.life_stress_plot.figure

                col1, col2 = st.columns(2)
                col1.write(probability_plot)
                col2.write(life_stress_plot)

                if use_level and hasattr(best_model, 'mean_life'):
                    st.write('# Use level analysis')
                    st.write(f'The mean life at use level {use_level} is **{best_model.mean_life:.4f}** time units.')


                    # if hasattr(best_model, 'change_of_parameters') and 'acceleration factor' in best_model.change_of_parameters.columns:
                    #     st.subheader("Acceleration Factor Analysis")

                    #     change_df = best_model.change_of_parameters.copy()
                    #     if pd.api.types.is_string_dtype(change_df['acceleration factor']):
                    #         change_df['acceleration factor'] = pd.to_numeric(change_df['acceleration factor'], errors='coerce')
                    #     change_df.dropna(subset=['acceleration factor'], inplace=True)

                    #     if not change_df.empty:

                    #         stress_cols_in_df = [col for col in change_df.columns if 'stress' in col.lower() and 'life-stress model' not in col.lower() and 'at use stress' not in col.lower()]

                    #         if not stress_cols_in_df and 'stress' in change_df.columns:
                    #             stress_cols_in_df = ['stress']
                    #         elif not stress_cols_in_df: #
                    #             st.caption("Could not identify stress column(s) in DataFrame 'change_of_parameters' for detailed display.")

                    #         cols_to_display = stress_cols_in_df + ['acceleration factor']


                    #         valid_cols_to_display = [col for col in cols_to_display if col in change_df.columns]

                    #         if valid_cols_to_display:
                    #             st.write("Acceleration Factors by Stress Level:")
                    #             st.dataframe(change_df[valid_cols_to_display].style.format({'acceleration factor': "{:.2f}x"}))
                    #         else:
                    #             st.write("Acceleration Factors (stress columns not found for drillthrough):")
                    #             st.dataframe(change_df[['acceleration factor']].style.format({'acceleration factor': "{:.2f}x"}))

                    #         metrics_data = []

                    #         avg_accel = change_df['acceleration factor'].mean()
                    #         metrics_data.append({"Metric": "Average Acceleration Factor", "Value": f"{avg_accel:.2f}x"})


                    #         if not change_df['acceleration factor'].empty:
                    #             max_accel_factor = change_df['acceleration factor'].max()
                    #             stress_at_max_accel_series = change_df.loc[change_df['acceleration factor'].idxmax(), [col for col in valid_cols_to_display if col != 'acceleration factor']]


                    #             if not stress_at_max_accel_series.empty:
                    #                 stress_at_max_accel_str = '; '.join(stress_at_max_accel_series.astype(str))
                    #                 metrics_data.append({
                    #                 "Metric": "Largest Acceleration Factor",
                    #                 "Value": f"{max_accel_factor:.2f}x (at stress: {stress_at_max_accel_str})"
                    #                 })
                    #             else:
                    #                 metrics_data.append({
                    #                 "Metric": "Largest Acceleration Factor",
                    #                 "Value": f"{max_accel_factor:.2f}x"
                    #                 })


                    #         if not change_df['acceleration factor'].empty:
                    #             min_accel_factor = change_df['acceleration factor'].min()
                    #             stress_at_min_accel_series = change_df.loc[change_df['acceleration factor'].idxmin(), [col for col in valid_cols_to_display if col != 'acceleration factor']]

                    #             if not stress_at_min_accel_series.empty:
                    #                 stress_at_min_accel_str = '; '.join(stress_at_min_accel_series.astype(str))
                    #                 metrics_data.append({
                    #                 "Metric": "Lowest Acceleration Factor",
                    #                 "Value": f"{min_accel_factor:.2f}x (at stress: {stress_at_min_accel_str})"
                    #                 })
                    #             else:
                    #                 metrics_data.append({
                    #                 "Metric": "Lowest Acceleration Factor",
                    #                 "Value": f"{min_accel_factor:.2f}x"
                    #                 })

                    #         num_stress_levels = len(change_df)
                    #         metrics_data.append({"Metric": "Number of Stress Levels with Acceleration Factor", "Value": num_stress_levels})

                    #         if metrics_data:
                    #             st.write("Summary of Acceleration Factor Metrics:")
                    #             metrics_df = pd.DataFrame(metrics_data)
                    #             st.table(metrics_df.set_index("Metric"))

                    #         else:
                    #             st.info("There is no valid acceleration factor data for quantitative analysis.")

                    #     elif hasattr(best_model, 'change_of_parameters'):
                    #         st.info("The 'acceleration factor' column was not found in the 'change_of_parameters' DataFrame. Acceleration factor analysis is not available.")

                    if hasattr(best_model, 'distribution_at_use_stress'):
                        st.write(f"""
                        #### Distribution at Use Stress Level
                        Characteristics of the best distribution at use conditions:
                        """)

                        dist = best_model.distribution_at_use_stress
                        try:
                            plot_params = functions.plot_parameters()
                            functions.plot_distribution(
                                    dist,
                                    plot_params,
                                    title=dist.param_title_long
                                )
                        except Exception as e:
                            st.warning(f"Could not plot distribution: {str(e)}")


    if st.session_state.final_censoring_type == "left":
        with st.expander('Short Guide'):
            st.write('When using this module, please take into consideration the following points:')
            st.info("""
            - There is no need to sort the data in any particular order.
            - For AFT models, ensure sufficient data points for model convergence, especially with multiple stress factors.
            - **Left-censored data** means the event (e.g., failure) is known to have occurred *before* the specified time.
            """)
            st.write("""
            This module uses `lifelines` for survival analysis:
            - **Kaplan-Meier:** Non-parametric estimation, supports left-censoring.
            - **Cox Proportional Hazards (CoxPH):** Semi-parametric model for hazard ratios. Left-censoring support may require specific baseline estimators (e.g., spline).
            - **Accelerated Failure Time (AFT) Models (Weibull, LogNormal, LogLogistic):**
            Parametric models where stresses are assumed to accelerate or decelerate the time to event.
            """)

        with st.expander("Equation Information"):
            st.write("For Parametric AFT models (Weibull, LogNormal, LogLogistic):")
            st.write(r"""
            The general form is:
            $$ \log(T) = \beta_0 + \beta_1 S_1 + \beta_2 S_2 + \dots + \sigma \epsilon $$
            Where:
            - $T$ is the time to event.
            - $S_1, S_2, \dots$ are the stress covariates.
            - $\beta_0, \beta_1, \beta_2, \dots$ are the regression coefficients.
            - $\sigma$ is a scale parameter.
            - $\epsilon$ is an error term from a specific distribution (e.g., Gumbel for Weibull AFT, Normal for LogNormal AFT).
            """)
            st.write("The `lifelines` library fits these coefficients.")
            st.write("For CoxPH model:")
            st.write(r"""
            The hazard function is modeled as:
            $$ h(t|X) = h_0(t) \exp(\beta_1 S_1 + \beta_2 S_2 + \dots) $$
            Where:
            - $h_0(t)$ is the baseline hazard function.
            - $\exp(\beta_i)$ is the hazard ratio for a unit change in stress $S_i$.
            """)
            st.write("For Kaplan-Meier model:")
            st.write("""
            The Kaplan-Meier (KM) estimator is a nonparametric method used to estimate the survival function from lifetime data.

            The survival function $S(t)$ is the probability of an individual surviving beyond time $t$. The Kaplan-Meier estimator is given by:
            """)
            st.latex(r'''
            \hat{S}(t) = \prod_{i: t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)
            ''')
            st.write(r"""
            Where:
            - $t_i$ are the distinct times in which at least one event (failure) occurred, ordered from smallest to largest.
            - $d_i$ is the number of events (failures) that occurred at time $t_i$.
            - $n_i$ is the number of individuals at risk (those who have not yet failed or been censored) just before time $t_i$.""")
            st.write(r""" **Hazard Function with Kaplan-Meier:**
            The Kaplan-Meier estimator focuses on the survival function $S(t)$. Although it does not provide a direct, smooth estimate of the hazard function $h(t)$, the hazard over an interval $[t_j, t_{j+1})$ can be approximated or estimated using methods such as the Nelson-Aalen estimator for the cumulative hazard function $\Lambda(t) = \int_0^t h(u)du$. From $\hat{\Lambda}(t)$, one can infer about $h(t)$. The Nelson-Aalen estimator for the cumulative hazard is:
            """)
            st.latex(r'''
            \hat{\Lambda}(t) = \sum_{i: t_i \leq t} \frac{d_i}{n_i}
            ''')
            st.write(r"""
            The instantaneous hazard function $h(t_i)$ at failure time $t_i$ can be thought of as the conditional "probability" of failure at $t_i$, given that it survived until $t_i$.
            """)
            st.markdown("---")

        df_lifelines = None
        dual_stress_model = False
        stress_cols = []

        if uploaded_file:
            df_input = pd.read_excel(uploaded_file)

            n_cols = len(df_input.columns)
            if n_cols < 3 or n_cols > 4:
                st.error('Invalid number of columns. Expected 3 or 4 columns: Time, Type, Stress1, [Stress2].')
                st.stop()

            col_names_map = {}
            base_names = ['Time', 'Type', 'Stress1']
            for i, name in enumerate(base_names):
                col_names_map[df_input.columns[i]] = name

            if n_cols == 4:
                col_names_map[df_input.columns[3]] = 'Stress2'
                dual_stress_model = True
                stress_cols = ['Stress1', 'Stress2']
            else:
                stress_cols = ['Stress1']

            df_input.rename(columns=col_names_map, inplace=True)
            col2_2.dataframe(df_input, width="stretch")

            df_lifelines = df_input.copy()

            type_mapping = {'F': 1, 'C': 0, 'L': -1}
            df_lifelines['Type'] = df_lifelines['Type'].astype(str).str.upper()
            if not all(item in type_mapping for item in df_lifelines['Type'].unique()):
                st.error(f"Invalid values in 'Type' column. Allowed values are 'F', 'C', 'L'. Found: {df_lifelines['Type'].unique()}")
                st.stop()
            df_lifelines['Event'] = df_lifelines['Type'].map(type_mapping)

            try:
                df_lifelines['Time'] = pd.to_numeric(df_lifelines['Time'])
                for stress_col in stress_cols:
                    df_lifelines[stress_col] = pd.to_numeric(df_lifelines[stress_col])
            except ValueError as e:
                st.error(f"Error converting columns to numeric. Please check data types. Details: {e}")
                st.stop()

            if df_lifelines[['Time', 'Event'] + stress_cols].isnull().any().any():
                st.error("Data contains NaN values after processing. Please check your input file, especially the 'Type' column and numeric conversions.")
                st.stop()

            st.markdown("---")
            st.subheader("Raw Data Timeline Visualization")
            if not df_lifelines.empty:
                fig_timeline, ax_timeline = plt.subplots(figsize=(10, max(4, len(df_lifelines) * 0.2)))

                durations_plot = df_lifelines['Time']
                event_observed_plot = (df_lifelines['Event'] == 1) # True para 'F', False para 'C' e 'L'

                plot_lifetimes(durations=durations_plot,
                            event_observed=event_observed_plot,
                            ax=ax_timeline)

                ax_timeline.set_xlabel("Time")
                ax_timeline.set_ylabel("Individual Observation")
                ax_timeline.set_title("Timeline of Individual Observations")
                plt.tight_layout()
                st.pyplot(fig_timeline)
                st.caption("""
                **Timeline Graph Interpretation:**
                - Each horizontal line represents an individual unit in your data set.
                - Lines start at time 0 and extend to the recorded time for that unit.
                - **Filled circles (●):** Indicate an observed failure (Type 'F') at that time.
                - **Open circles (○):** Indicate a censored observation at that time. For this graph, both **Right-Censored (Type 'C')** and **Left-Censored (Type 'L')** data are shown with an open circle.
                - For 'C', the event *did not* occur until that time.
                - For 'L', the event occurred *before* that time.
                """)
            else:
                st.info("No data available to display timeline.")
            st.markdown("---")

            use_level_input = st.text_input("Use level stress (optional, for AFT/CoxPH predictions)")
            use_level_stresses = None
            if use_level_input:
                try:
                    use_level_parts = use_level_input.strip().split(sep=';')
                    use_level_stresses = [float(x.strip()) for x in use_level_parts]
                    if dual_stress_model and len(use_level_stresses) != 2:
                        st.error('Dual stress model: Please enter two use level stresses separated by ";"!')
                        st.stop()
                    elif not dual_stress_model and len(use_level_stresses) != 1:
                        st.error('Single stress model: Please enter one use level stress only!')
                        st.stop()
                    if len(use_level_stresses) != len(stress_cols):
                        st.error(f"Number of use level stresses ({len(use_level_stresses)}) does not match number of stress factors ({len(stress_cols)}).")
                        st.stop()

                except ValueError:
                    st.error("Invalid format for use level stress. Enter numeric values, separated by ';' for dual stress.")
                    st.stop()

            available_models = list(LIFELINES_MODELS.keys())
            selected_models = st.multiselect('Choose which model(s) you want to fit:', available_models)

        if st.button("Fit Model(s)"):
            if not uploaded_file or df_lifelines is None:
                st.error('Please upload a file and ensure it is processed correctly!')
                st.stop()
            if not selected_models:
                st.error("Please select at least one model to fit.")
                st.stop()

            with st.spinner('Fitting models...'):
                st.write("## Fitting Results")

                all_results_summary = []
                best_model_instance = None
                best_model_name = ""
                best_aic = np.inf

                for model_name in selected_models:
                    st.subheader(f"Model: {model_name}")
                    fitter_class = LIFELINES_MODELS[model_name]

                    if model_name == "CoxPH":
                        fitter = CoxPHFitter(baseline_estimation_method="spline",n_baseline_knots=4)
                    else:
                        fitter = fitter_class()

                    try:
                        if model_name == "Kaplan-Meier":
                            fitter.fit_left_censoring(df_lifelines['Time'], df_lifelines['Event'])
                            st.write("Kaplan-Meier fitted on pooled data.")
                            fig, ax = plt.subplots()
                            fitter.plot_survival_function(ax=ax)
                            ax.set_title(f"Kaplan-Meier Survival Function (Left Censored Data)")
                            st.pyplot(fig)
                            st.write("Survival function estimates:")
                            st.dataframe(fitter.survival_function_)
                            all_results_summary.append({
                                "Model": model_name, "AIC": "N/A", "Log-Likelihood": "N/A",
                                "Converged": "N/A", "Parameters": "N/A (Non-parametric)"
                            })

                        elif model_name == "CoxPH":
                            cols_for_cox = ['Time', 'Event'] + stress_cols
                            df_cox_fit = df_lifelines[cols_for_cox].copy()

                            if not stress_cols:
                                st.warning("CoxPH: No stress covariates identified. Model will estimate baseline hazard only using spline method.")

                            fitter.fit_left_censoring(df_cox_fit, 'Time', 'Event')

                            fitter.print_summary(decimals=4)
                            summary_df = fitter.summary
                            st.dataframe(summary_df)

                            log_lik = fitter.log_likelihood_ if hasattr(fitter, 'log_likelihood_') else 'N/A'
                            aic_val = fitter.AIC_ if hasattr(fitter, 'AIC_') else 'N/A'
                            converged = fitter.convergence_flags_['converged'] if hasattr(fitter, 'convergence_flags_') and fitter.convergence_flags_ else 'N/A'

                            all_results_summary.append({
                                "Model": model_name, "AIC": f"{aic_val:.2f}" if isinstance(aic_val, float) else aic_val,
                                "Log-Likelihood": f"{log_lik:.2f}" if isinstance(log_lik, float) else log_lik,
                                "Converged": converged, "Parameters": "See Summary Table"
                            })
                            if isinstance(aic_val, float) and aic_val < best_aic:
                                best_aic = aic_val
                                best_model_instance = fitter
                                best_model_name = model_name

                            # Plotting for CoxPH
                            fig, ax = plt.subplots()
                            if stress_cols and use_level_stresses:
                                df_use_level_cox = pd.DataFrame([use_level_stresses], columns=stress_cols)
                                try:
                                    sf_prediction = fitter.predict_survival_function(df_use_level_cox)
                                    sf_prediction.rename(columns={0: f"Use Level: {use_level_stresses}"}).plot(ax=ax)
                                    ax.set_title(f"CoxPH Predicted Survival at Use Level")
                                except Exception as plot_e:
                                    st.warning(f"Could not plot predicted survival for CoxPH at use level: {plot_e}. Plotting baseline.")
                                    fitter.plot_baseline_survival(ax=ax)
                                    ax.set_title(f"CoxPH Baseline Survival Function")
                            elif stress_cols:
                                X_mean_cox = df_lifelines[stress_cols].mean().to_frame().T
                                try:
                                    sf_prediction_mean = fitter.predict_survival_function(X_mean_cox)
                                    sf_prediction_mean.rename(columns={0: "Avg Stress"}).plot(ax=ax)
                                    ax.set_title(f"CoxPH Predicted Survival (Avg Stress)")
                                except Exception as plot_e:
                                    st.warning(f"Could not plot predicted survival for CoxPH at avg stress: {plot_e}. Plotting baseline.")
                                    fitter.plot_baseline_survival(ax=ax)
                                    ax.set_title(f"CoxPH Baseline Survival Function")
                            else:
                                fitter.plot_baseline_survival(ax=ax)
                                ax.set_title(f"CoxPH Baseline Survival Function")
                            st.pyplot(fig)


                        elif "AFT" in model_name: # WeibullAFT, LogNormalAFT, LogLogisticAFT
                            df_aft_fit = df_lifelines[['Time', 'Event'] + stress_cols].copy()

                            if not stress_cols:
                                st.warning(f"{model_name} is typically used with stress covariates. Fitting as a simple distribution (no covariates).")
                                fitter.fit(df_aft_fit['Time'], event_observed=df_aft_fit['Event'])
                            else:
                                formula = " + ".join(stress_cols)
                                fitter.fit(df_aft_fit, 'Time', 'Event', formula=formula)

                            fitter.print_summary(decimals=4)
                            summary_df = fitter.summary
                            st.dataframe(summary_df)

                            log_lik = fitter.log_likelihood_
                            aic_val = fitter.AIC_
                            converged_status  ='N/A'

                            if hasattr(fitter, '_CONVERGENCE_FLAG_KEY'):
                                if fitter._CONVERGENCE_FLAG_KEY in fitter.convergence_flags_:
                                    converged_status = fitter.convergence_flags_[fitter._CONVERGENCE_FLAG_KEY]
                                elif hasattr(fitter, 'params_'):
                                    converged_status = True
                            elif hasattr(fitter, 'params_'):
                                converged_status = True

                            all_results_summary.append({
                                "Model": model_name, "AIC": f"{aic_val:.2f}",
                                "Log-Likelihood": f"{log_lik:.2f}",
                                "Converged": converged_status,
                                "Parameters": "N/A"
                            })

                            if isinstance(aic_val, (float, int)) and aic_val < best_aic :
                                best_aic = aic_val
                                best_model_instance = fitter
                                best_model_name = model_name

                            # Plotting for AFT
                            fig, ax = plt.subplots()
                            if stress_cols and use_level_stresses:
                                df_use_level_aft = pd.DataFrame([use_level_stresses], columns=stress_cols)
                                fitter.predict_survival_function(df_use_level_aft).rename(columns={0: f"Use Level: {use_level_stresses}"}).plot(ax=ax)
                                ax.set_title(f"{model_name} Survival Function at Use Level")
                            elif stress_cols:
                                X_mean_aft = df_lifelines[stress_cols].mean().to_frame().T
                                fitter.predict_survival_function(X_mean_aft).rename(columns={0: "Avg Stress"}).plot(ax=ax)
                                ax.set_title(f"{model_name} Survival Function (Avg Stress)")
                            else:
                                fitter.plot_survival_function(ax=ax)
                                ax.set_title(f"{model_name} Survival Function (No Covariates)")
                            st.pyplot(fig)

                    except (ConvergenceError, RuntimeError, ValueError, np.linalg.LinAlgError) as e:
                        st.error(f"Could not fit {model_name}. Error: {e}")
                        all_results_summary.append({
                            "Model": model_name, "AIC": "Error", "Log-Likelihood": "Error",
                            "Converged": "Error", "Parameters": f"Error: {e}"
                        })
                    except Exception as e:
                        st.error(f"An unexpected error occurred while fitting {model_name}: {e}")
                        all_results_summary.append({
                            "Model": model_name, "AIC": "Error", "Log-Likelihood": "Error",
                            "Converged": "Error", "Parameters": f"Error: {e}"
                        })


                st.write("## Summary of All Fitted Models")
                if all_results_summary:
                    results_df = pd.DataFrame(all_results_summary)
                    #st.dataframe(results_df) ##ERRO AQUI
                    st.write(all_results_summary)
                    if best_model_instance and best_model_name and best_aic != np.inf :
                        st.write(f"## Best Model: {best_model_name}")
                        #st.write(f"AIC: {best_aic:.2f}")
                        #st.write("Summary of Best Model:")
                        #st.dataframe(best_model_instance.summary)

                        if use_level_stresses and hasattr(best_model_instance, 'predict_median') and stress_cols:
                            if "AFT" in best_model_name or "CoxPH" in best_model_name :
                                df_predict_use_level = pd.DataFrame([use_level_stresses], columns=stress_cols)
                                try:
                                    median_life_prediction = best_model_instance.predict_median(df_predict_use_level)

                                    actual_median_life_value = None
                                    if isinstance(median_life_prediction, (pd.Series, pd.DataFrame)):
                                        actual_median_life_value = median_life_prediction.iloc[0]
                                    elif isinstance(median_life_prediction, (float, np.float64, int, np.int64)):
                                        actual_median_life_value = median_life_prediction
                                    else:
                                        st.warning(f"Unexpected type for median life prediction: {type(median_life_prediction)}")

                                    if actual_median_life_value is not None:
                                        st.write(f"Predicted Median Life at Use Level {use_level_stresses}: **{actual_median_life_value:.2f}** time units.")

                                except Exception as e:
                                    st.warning(f"Could not predict for use level with best model. Error: {e}")
                        elif use_level_stresses and not stress_cols and hasattr(best_model_instance, 'predict_median'):
                            st.info("Use level stresses provided, but the best model was fit without stress covariates. Predictions at use level are not applicable in this context for this model.")


                    elif not best_model_name and any("Error" not in r["AIC"] for r in all_results_summary if isinstance(r["AIC"], str)):
                        st.info("No model was successfully chosen as 'best' based on AIC, possibly due to errors or all models having N/A AIC values.")
                    else:
                        st.warning("No models were successfully fitted or no suitable best model found.")
                else:
                    st.info("No models were selected or processed.")

    if st.session_state.final_censoring_type == "interval":
        with st.expander('Short Guide'):
            st.write('When using this module, please take into consideration the following points:')
            st.info("""
            - There is no need to sort the data in any particular order.
            - For AFT models, ensure sufficient data points for model convergence, especially with multiple stress factors.
            - **Interval-censored data** means the event (e.g., failure) is known to have occurred *between* a lower and an upper time bound.
            """)
            st.write("""
            This module uses `lifelines` for survival analysis with interval-censored data:
            - **Kaplan-Meier (Turnbull estimator):** A non-parametric method to estimate the survival function from interval-censored data.
            - **Accelerated Failure Time (AFT) Models (Weibull, LogNormal, LogLogistic):**
            Parametric models where stresses are assumed to accelerate or decelerate the time to event.
            """)

        with st.expander("Equation Information"):
            st.write("#### Kaplan-Meier (Turnbull Estimator)")
            st.write("""
            For interval-censored data, the Kaplan-Meier estimator is generalized using the Turnbull algorithm. This is a non-parametric method that computes the Non-Parametric Maximum Likelihood Estimator (NPMLE) of the survival function. It does not assume an underlying distribution for the lifetime but finds a discrete survival function that best fits the observed failure intervals. The algorithm iteratively estimates the survival probabilities over the disjoint intervals derived from the data.
            """)

            st.write("#### Accelerated Failure Time (AFT) Models")
            st.write(r"""
            The general form of an AFT model is:
            $$ \log(T) = \beta_0 + \beta_1 S_1 + \beta_2 S_2 + \dots + \sigma \epsilon $$
            For an interval-censored observation $(L, R)$, the contribution to the likelihood function is the probability of the failure occurring within that interval:
            $$ P(L < T \le R | \mathbf{S}) = P(\log(L) < \log(T) \le \log(R) | \mathbf{S}) $$
            This is equivalent to $F(\log(R)|\mathbf{S}) - F(\log(L)|\mathbf{S})$, or in terms of the survival function, $S(L|\mathbf{S}) - S(R|\mathbf{S})$, where $S$ is the survival function of the chosen AFT model (e.g., Weibull, LogNormal). The model's parameters are estimated by maximizing the product of these probabilities across all observations.
            """)

        df_lifelines = None
        dual_stress_model = False
        stress_cols = []
        use_level_stresses = None

        if uploaded_file:
            try:
                df_input = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading Excel file: {e}")
                st.stop()

            n_cols = len(df_input.columns)
            expected_min_cols, expected_max_cols = 4, 5
            if n_cols < expected_min_cols or n_cols > expected_max_cols:
                st.error(f'Expected {expected_min_cols} or {expected_max_cols} columns. Found {n_cols}: {list(df_input.columns)}')
                st.stop()

            col_map = {df_input.columns[0]: 'Time_Lower', df_input.columns[1]: 'Time_Upper',
                    df_input.columns[2]: 'Event', df_input.columns[3]: 'Stress1'}
            stress_cols = ['Stress1']
            if n_cols == expected_max_cols:
                col_map[df_input.columns[4]] = 'Stress2'
                dual_stress_model = True
                stress_cols.append('Stress2')

            df_lifelines = df_input.rename(columns=col_map)
            col2_2.dataframe(df_lifelines.head(), width="stretch")

            try:
                df_lifelines['Event'] = pd.to_numeric(df_lifelines['Event'], errors='coerce')
                if df_lifelines['Event'].isnull().any() or not df_lifelines['Event'].isin([0, 1]).all():
                    st.error("Data Error: Event column must be 0 or 1 and contain no missing values after conversion.")
                    st.stop()

                df_lifelines['Time_Lower'] = pd.to_numeric(df_lifelines['Time_Lower'], errors='coerce').fillna(0)
                df_lifelines['Time_Upper'] = pd.to_numeric(df_lifelines['Time_Upper'], errors='coerce')
                df_lifelines.loc[(df_lifelines['Event'] == 0) & (df_lifelines['Time_Upper'].isnull()), 'Time_Upper'] = np.inf

                is_event_1 = df_lifelines['Event'] == 1
                is_time_upper_nan = df_lifelines['Time_Upper'].isnull()
                is_event_at_infinity = np.isinf(df_lifelines['Time_Lower']) & \
                                    np.isinf(df_lifelines['Time_Upper']) & \
                                    (df_lifelines['Time_Lower'] == df_lifelines['Time_Upper']) & \
                                    is_event_1

                if (is_event_1 & is_time_upper_nan & ~is_event_at_infinity).any():
                    st.error("Data Error: Found rows with Event=1 where Time_Upper is missing (NaN/blank). "
                            "For left-censored or interval-censored events, Time_Upper must be a finite number. "
                            "For exact events, Time_Upper should equal Time_Lower (and be finite). "
                            "Please correct your input data.")
                    st.stop()

                condition_event1_upper_inf_not_both_inf = is_event_1 & np.isinf(df_lifelines['Time_Upper']) & ~is_event_at_infinity
                if condition_event1_upper_inf_not_both_inf.any():
                    st.warning("Data Warning: Found rows with Event=1, finite/zero Time_Lower, but Time_Upper is infinity. "
                            "This implies the event is observed but at an unbounded future time. "
                            "This is unusual and might be an error. For AFT models using fit_interval_censoring, "
                            "such data might be treated as censored (event_observed=0) because Time_Lower != Time_Upper.")


                for sc in stress_cols:
                    df_lifelines[sc] = pd.to_numeric(df_lifelines[sc], errors='coerce')

                if (df_lifelines['Time_Lower'] > df_lifelines['Time_Upper']).any():
                    st.error("Data Error: Time_Lower > Time_Upper found in some rows. Please correct your input data.")
                    st.stop()

                df_lifelines = df_lifelines[~(np.isinf(df_lifelines['Time_Lower']) & ~is_event_at_infinity)]

                df_lifelines['Event_Turnbull'] = np.where(
                    (df_lifelines['Time_Lower'] == df_lifelines['Time_Upper']) & np.isfinite(df_lifelines['Time_Lower']),
                    1, 0
                )
                df_lifelines.loc[df_lifelines['Event'] == 0, 'Event_Turnbull'] = 0

                df_lifelines['Event_AFT_IntervalCensoring'] = np.where(
                    (df_lifelines['Time_Lower'] == df_lifelines['Time_Upper']) & np.isfinite(df_lifelines['Time_Upper']),
                    1, 0
                )

            except Exception as e:
                st.error(f"Error during data processing. Details: {e}")
                st.error(traceback.format_exc())
                st.stop()

            if df_lifelines.empty:
                st.error("Data Error: DataFrame is empty after initial processing/filtering. Please check your input data.")
                st.stop()

            #st.write("### Processed Data Snippet")
            #st.dataframe(df_lifelines[['Time_Lower', 'Time_Upper', 'Event'] + stress_cols].head())


            st.write("### Raw Interval Data Visualization")
            try:
                fig_height = max(3, min(10, len(df_lifelines) * 0.15 if len(df_lifelines) > 0 else 3))
                fig_raw, ax_raw = plt.subplots(figsize=(10, fig_height))
                if not df_lifelines.empty:
                    lifelines_plotting.plot_interval_censored_lifetimes(
                        df_lifelines['Time_Lower'],
                        df_lifelines['Time_Upper'],
                        event_observed=df_lifelines['Event'],
                        ax=ax_raw
                    )
                ax_raw.set_xlabel("Time"); ax_raw.set_ylabel("Observation Index")
                ax_raw.set_title("Visualization of Interval Censored Lifetimes (User 'Event' for color)")
                st.pyplot(fig_raw)
                st.caption("""
                **Interpreting the Interval Plot:**
                - Each horizontal line represents an individual unit or observation.
                - **Red lines (with markers):** Indicate that an event (failure) was observed (`Event=1`).
                    - A line from one point to another (e.g., `>-----<`) shows an **interval of uncertainty**. The failure occurred at some point between the start and end of the line.
                    - A solid dot (`●`) at an exact time signifies an **exact failure** (`Time_Lower` == `Time_Upper`).
                    - A line starting at 0 and ending with a marker at T (`>-----|`) represents **left-censorship** (failure occurred *before* time T).
                - **Blue lines (without end markers):** Indicate **right-censored** observations (`Event=0`). The event had not occurred by the end of the observation period, which is the right boundary of the line.
                """)
            except Exception as e:
                st.warning(f"Could not plot raw interval data: {e}")
                st.warning(traceback.format_exc())


            use_level_input_value = st.text_input("Use level stress (optional, for AFT/CoxPH predictions)")
            if use_level_input_value:
                try:
                    use_level_parts = use_level_input_value.strip().split(sep=';')
                    use_level_stresses = [float(x.strip()) for x in use_level_parts]
                    if dual_stress_model and len(use_level_stresses) != 2:
                        st.error('Dual stress model: Please enter two use level stresses separated by ";"!')
                        use_level_stresses = None; st.stop()
                    elif not dual_stress_model and len(use_level_stresses) != 1:
                        st.error('Single stress model: Please enter one use level stress only!')
                        use_level_stresses = None; st.stop()
                    if stress_cols and (len(use_level_stresses) != len(stress_cols)):
                        st.error(f"Number of use level stresses ({len(use_level_stresses)}) does not match number of stress factors ({len(stress_cols)}).")
                        use_level_stresses = None; st.stop()
                except ValueError:
                    st.error("Invalid format for use level stress. Enter numeric values, separated by ';' for dual stress.")
                    use_level_stresses = None; st.stop()


        available_models_interval = list(LIFELINES_MODELS.keys())
        selected_models = st.multiselect('Choose which model(s) you want to fit:', available_models_interval)

        cox_params_user = {}
        if "CoxPH" in selected_models:
            st.markdown("---")
            st.write("Please, input CoxPH Parameters")

            cox_params_user['alpha'] = st.number_input(
                "Alpha (for CIs)", min_value=0.01, max_value=0.99, value=0.95, step=0.01,
                help="Significance level for confidence intervals of coefficients (e.g., 0.95 for 95% CI).",
                key="cox_alpha"
            )
            cox_params_user['penalizer'] = st.number_input(
                "Penalizer (L2)", min_value=0.0, value=0.0, step=0.01, format="%.4f",
                help="L2 penalty. 0.0 = no penalty. Small positive values (e.g., 0.01, 0.1) can aid convergence.",
                key="cox_penalizer"
            )
            cox_params_user['n_baseline_knots'] = st.number_input(
                "Number of Baseline Knots", min_value=2, max_value=20, value=3, step=1,
                help="Number of knots for spline baseline. More knots = more flexible baseline.",
                key="cox_knots"
            )


        if st.button("Fit Model(s)"):
            if not uploaded_file or df_lifelines is None or df_lifelines.empty:
                st.error('Please upload a file. Ensure it is processed correctly and is not empty after validation!')
                st.stop()
            if not selected_models:
                st.error("Please select at least one model to fit.")
                st.stop()

            with st.spinner("Fitting selected models... Please wait."):
                st.write("## Fitting Results")
                all_results_summary = []

                for model_name in selected_models:
                    st.subheader(f"Model: {model_name}")
                    fitter_class = LIFELINES_MODELS[model_name]
                    fitter = None

                    try:
                        essential_cols_km = ['Time_Lower', 'Time_Upper', 'Event_Turnbull']
                        essential_cols_cox = ['Time_Lower', 'Time_Upper', 'Event_Turnbull'] + stress_cols
                        essential_cols_aft_interval = ['Time_Lower', 'Time_Upper', 'Event_AFT_IntervalCensoring'] + stress_cols
                        df_fit = df_lifelines.copy()

                        if "Kaplan-Meier" in model_name:
                            fitter = fitter_class()
                            df_fit_km = df_fit[essential_cols_km].dropna(subset=['Time_Lower', 'Time_Upper'])
                            if df_fit_km.empty:
                                st.warning(f"KM: Data empty. Skipping.")
                                continue
                            fitter.fit_interval_censoring(
                                lower_bound=df_fit_km['Time_Lower'],
                                upper_bound=df_fit_km['Time_Upper'],
                                event_observed=df_fit_km['Event_Turnbull']
                            )
                            st.write("Kaplan-Meier fitted on pooled data.")
                            fig, ax = plt.subplots()
                            st.write("Survival function estimates:"); st.dataframe(fitter.survival_function_)


                            survival_df_km_interval = fitter.survival_function_

                            if 'NPMLE_estimate_lower' in survival_df_km_interval.columns and \
                               'NPMLE_estimate_upper' in survival_df_km_interval.columns:

                                ax.step(survival_df_km_interval.index, survival_df_km_interval['NPMLE_estimate_upper'],
                                where='post', label='Upper Limit of Estimate (NPMLE)', color='blue', alpha=0.7)
                                ax.step(survival_df_km_interval.index, survival_df_km_interval['NPMLE_estimate_lower'],
                                where='post', label='Lower Limit of Estimation (NPMLE)', color='green', alpha=0.7)

                                ax.fill_between(survival_df_km_interval.index,
                                survival_df_km_interval['NPMLE_estimate_lower'],
                                survival_df_km_interval['NPMLE_estimate_upper'],
                                step='post', alpha=0.2, color='gray',
                                label=f'Confidence Interval ({100*(1-fitter.alpha):.0f}%)')

                                ax.set_xlabel("Time")
                                ax.set_ylabel("Survival Probability S(t)")
                                ax.set_title(f"Kaplan-Meier Survival Function (Interval)")
                                ax.legend()
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                fitter.plot_survival_function(ax=ax)
                                ax.set_title(f"Kaplan-Meier")
                                st.pyplot(fig)

                            st.write("The estimated median time to event:")
                            st.write( fitter.median_survival_time_)

                        elif "CoxPH" in model_name:
                            current_cox_args = {
                                'alpha': cox_params_user.get('alpha', 0.95),
                                'penalizer': cox_params_user.get('penalizer', 0.0),
                                'baseline_estimation_method': cox_params_user.get('baseline_estimation_method', 'spline')
                            }
                            if current_cox_args['baseline_estimation_method'] == 'spline':
                                current_cox_args['n_baseline_knots'] = cox_params_user.get('n_baseline_knots', 3)
                            fitter = CoxPHFitter(**current_cox_args)
                            df_fit_cox = df_fit[essential_cols_cox].dropna(subset=['Time_Lower', 'Time_Upper'] + stress_cols if stress_cols else ['Time_Lower', 'Time_Upper'])
                            if df_fit_cox.empty:
                                st.warning(f"CoxPH: Data empty. Skipping.")
                                all_results_summary.append({"Model": model_name, "AIC": "Data insufficient", "Log-Likelihood": "Data insufficient", "Converged": "N/A", "Median Survival": "N/A"})
                                continue
                            if not stress_cols: st.warning("CoxPH: No stress covariates.")

                            try:
                                fitter.fit_interval_censoring(df_fit_cox, 'Time_Lower', 'Time_Upper', event_col='Event_Turnbull')
                                st.success(f"{model_name} fitted successfully!")
                                fitter.print_summary(decimals=4); summary_df = fitter.summary; st.dataframe(summary_df)
                                log_lik = fitter.log_likelihood_; aic_val = fitter.AIC_
                                converged_status_cox = getattr(fitter, 'convergence_flags_', {}).get('converged', True)

                                median_cox_val_str = "N/A"
                                if stress_cols and use_level_stresses:
                                    df_pred_cox = pd.DataFrame([use_level_stresses], columns=stress_cols)
                                    try:
                                        median_cox_pred = fitter.predict_median(df_pred_cox)
                                        median_cox_val = median_cox_pred.iloc[0] if isinstance(median_cox_pred, pd.Series) else median_cox_pred
                                        st.write(f"**Predicted Median Survival Time (CoxPH at Use Level {use_level_stresses}):** {median_cox_val:.2f}" if not np.isnan(median_cox_val) else f"**Predicted Median Survival Time (CoxPH at Use Level {use_level_stresses}):** Not reached or N/A")
                                        median_cox_val_str = f"{median_cox_val:.2f}" if not np.isnan(median_cox_val) else "N/A"
                                    except Exception as e_m_cox:
                                        st.caption(f"Could not predict median for CoxPH at use level: {e_m_cox}")
                                elif not stress_cols:
                                    try:
                                        st.write(f"**Median Survival Time (CoxPH Baseline):** N/A (Median typically predicted for specific covariate values)")
                                    except Exception as e_m_base_cox:
                                        st.caption(f"Note on CoxPH baseline median: {e_m_base_cox}")


                                all_results_summary.append({"Model": model_name, "AIC": f"{aic_val:.2f}", "Log-Likelihood": f"{log_lik:.2f}", "Converged": converged_status_cox, "Median Survival": median_cox_val_str})
                                if converged_status_cox is True:
                                    fig_cox, ax_cox = plt.subplots()

                                    plot_title_cox = f"CoxPH Survival"
                                    prediction_context_cox = ""

                                    if stress_cols and use_level_stresses:
                                        df_use_level_cox = pd.DataFrame([use_level_stresses], columns=stress_cols)
                                        try:
                                            survival_function_pred = fitter.predict_survival_function(df_use_level_cox)
                                            survival_function_pred.iloc[:, 0].rename(f"Use: {use_level_stresses}").plot(ax=ax_cox)
                                            prediction_context_cox = f"at Use Level {use_level_stresses}"
                                        except Exception as e_plot_use:
                                            st.warning(f"Error predicting/plotting survival for CoxPH at use level: {e_plot_use}. Plotting baseline survival.")
                                            if hasattr(fitter, 'baseline_survival_'):
                                                fitter.plot_baseline_survival(ax=ax_cox)
                                                prediction_context_cox = "(Baseline due to Prediction Error)"
                                            else:
                                                ax_cox.text(0.5, 0.5, "Baseline survival not available for plotting.", horizontalalignment='center', verticalalignment='center', transform=ax_cox.transAxes)
                                                prediction_context_cox = "(Plotting Error)"
                                    elif stress_cols:
                                        df_mean_stress_cox = df_fit_cox[stress_cols].mean().to_frame().T
                                        if not df_mean_stress_cox.isnull().values.any():
                                            try:
                                                survival_function_mean = fitter.predict_survival_function(df_mean_stress_cox)
                                                survival_function_mean.iloc[:, 0].rename("Avg Stress").plot(ax=ax_cox)
                                                prediction_context_cox = f"at Average Stress ({df_mean_stress_cox.iloc[0].to_dict()})"
                                            except Exception as e_plot_mean:
                                                st.warning(f"Error predicting/plotting survival for CoxPH at average stress: {e_plot_mean}. Plotting baseline survival.")
                                                if hasattr(fitter, 'baseline_survival_'):
                                                    fitter.plot_baseline_survival(ax=ax_cox)
                                                    prediction_context_cox = "(Baseline due to Prediction Error at Avg Stress)"
                                                else:
                                                    ax_cox.text(0.5, 0.5, "Baseline survival not available for plotting.", horizontalalignment='center', verticalalignment='center', transform=ax_cox.transAxes)
                                                    prediction_context_cox = "(Plotting Error)"
                                        else:
                                            st.caption("Plotting baseline survival for CoxPH as average stress could not be determined.")
                                            if hasattr(fitter, 'baseline_survival_'):
                                                fitter.plot_baseline_survival(ax=ax_cox)
                                                prediction_context_cox = "(Baseline - Avg Stress Undetermined)"
                                            else:
                                                ax_cox.text(0.5, 0.5, "Baseline survival not available for plotting.", horizontalalignment='center', verticalalignment='center', transform=ax_cox.transAxes)
                                                prediction_context_cox = "(Plotting Error)"
                                    else:
                                        if hasattr(fitter, 'baseline_survival_'):
                                            fitter.plot_baseline_survival(ax=ax_cox)
                                            prediction_context_cox = "(Baseline Survival)"
                                        else:
                                            ax_cox.text(0.5, 0.5, "Baseline survival not available for plotting.", horizontalalignment='center', verticalalignment='center', transform=ax_cox.transAxes)
                                            prediction_context_cox = "(Plotting Error)"

                                    ax_cox.set_title(f"{plot_title_cox} {prediction_context_cox}")
                                    ax_cox.set_xlabel("Time")
                                    ax_cox.set_ylabel("Survival Probability S(t)")
                                    st.pyplot(fig_cox)
                                else:
                                    st.caption("Plotting skipped for CoxPH model as it did not converge or an error occurred during fitting.")

                            except ConvergenceError:
                                st.error(f"The {model_name} model did not converge.")
                                all_results_summary.append({"Model": model_name, "AIC": "Convergence Error", "Log-Likelihood": "Convergence Error", "Converged": False, "Median Survival": "N/A"})
                                explain_convergence_error()

                        elif "AFT" in model_name:
                            fitter = fitter_class()
                            df_fit_aft = df_fit[essential_cols_aft_interval].dropna(subset=['Time_Lower', 'Time_Upper'] + stress_cols if stress_cols else ['Time_Lower', 'Time_Upper'])
                            if df_fit_aft.empty:
                                st.warning(f"AFT ({model_name}): Data empty. Skipping.")
                                all_results_summary.append({"Model": model_name, "AIC": "Data insufficient", "Log-Likelihood": "Data insufficient", "Converged": "N/A", "Median Survival": "N/A", "MTTF": "N/A"})
                                continue

                            try:
                                if not stress_cols:
                                    fitter.fit_interval_censoring(durations=(df_fit_aft['Time_Lower'], df_fit_aft['Time_Upper']), event_observed=df_fit_aft['Event_AFT_IntervalCensoring'])
                                else:
                                    formula = " + ".join(stress_cols)
                                    fitter.fit_interval_censoring(df_fit_aft, 'Time_Lower', 'Time_Upper', event_col='Event_AFT_IntervalCensoring', formula=formula)

                                st.success(f"{model_name} fitted successfully!")
                                fitter.print_summary(decimals=4); summary_df = fitter.summary; st.dataframe(summary_df)
                                log_lik = fitter.log_likelihood_; aic_val = fitter.AIC_
                                converged_status_aft = getattr(fitter, 'convergence_flags_', {}).get('converged', True)

                                st.write("**AFT Model Predictions:**")
                                percentiles_to_predict = [0.05, 0.10,0.25, 0.50,0.75, 0.90]
                                df_for_prediction_aft = None
                                prediction_label_aft = "Baseline"
                                median_aft_val_str = "N/A"
                                mttf_aft_val_str = "N/A"

                                if stress_cols:
                                    prediction_label_aft = "Average Stress"
                                    if use_level_stresses:
                                        df_for_prediction_aft = pd.DataFrame([use_level_stresses], columns=stress_cols)
                                        prediction_label_aft = f"Use Level: {use_level_stresses}"
                                    else:
                                        df_mean_stress_aft = df_fit_aft[stress_cols].mean().to_frame().T
                                        if not df_mean_stress_aft.isnull().values.any():
                                            df_for_prediction_aft = df_mean_stress_aft

                                percentile_predictions_list = []
                                if hasattr(fitter, 'predict_percentile'):
                                    st.write(f"**Predicted Percentiles for {model_name} ({prediction_label_aft}):**")
                                    for p_aft in percentiles_to_predict:
                                        percentile_label = f"B{p_aft*100:.0f} (Percentile {p_aft*100:.0f}%)"
                                        display_val_aft = "Forecast Error"
                                        try:
                                            pred_val_aft = fitter.predict_percentile(
                                                df_for_prediction_aft, p=p_aft
                                            ) if df_for_prediction_aft is not None else fitter.predict_percentile(p=p_aft)

                                            if isinstance(pred_val_aft, (pd.Series, pd.DataFrame)):
                                                numeric_val_aft = pred_val_aft.iloc[0,0] if isinstance(pred_val_aft, pd.DataFrame) else pred_val_aft.iloc[0]
                                            elif isinstance(pred_val_aft, (np.ndarray)) and pred_val_aft.ndim > 0:
                                                numeric_val_aft = pred_val_aft.item(0) if pred_val_aft.size == 1 else pred_val_aft[0]
                                            else:
                                                numeric_val_aft = pred_val_aft


                                            if not pd.isna(numeric_val_aft):
                                                display_val_aft = f"{numeric_val_aft:.2f}"
                                                if p_aft == 0.50:
                                                    median_aft_val_str = display_val_aft
                                            else:
                                                display_val_aft = "NaN"

                                        except Exception as e_perc:
                                            st.caption(f"Error calculating {percentile_label}: {e_perc}")

                                        percentile_predictions_list.append({
                                            "Percentile (Life B)": percentile_label,
                                            "Estimated Time": display_val_aft
                                        })

                                    if percentile_predictions_list:
                                        percentiles_df = pd.DataFrame(percentile_predictions_list)
                                        st.table(percentiles_df.set_index("Percentile (Life B)"))

                                    else:
                                        st.caption("No percentile could be predicted.")

                                if hasattr(fitter, 'predict_expectation'):
                                    try:
                                        expected_life_aft = fitter.predict_expectation(df_for_prediction_aft) if df_for_prediction_aft is not None else fitter.predict_expectation()
                                        numeric_el_aft = expected_life_aft.iloc[0] if isinstance(expected_life_aft, pd.Series) else expected_life_aft
                                        mttf_aft_val_str = f"{numeric_el_aft:.2f}" if not np.isnan(numeric_el_aft) else "NaN"
                                        st.write(f"Predicted MTTF ({prediction_label_aft}): **{mttf_aft_val_str}**" if mttf_aft_val_str != "NaN" else f"Predicted MTTF ({prediction_label_aft}): **NaN** (May not be defined/computable)")
                                    except NotImplementedError:
                                        mttf_aft_val_str = "Not Implemented"
                                        st.write(f"Predicted MTTF for {model_name} ({prediction_label_aft}): Not implemented.")
                                    except Exception:
                                        mttf_aft_val_str = "Error"
                                        st.write(f"Could not calculate Expected Life for {model_name} ({prediction_label_aft}).")
                                elif "LogLogistic" in model_name:
                                    mttf_aft_val_str = "May not be defined"
                                    st.write(f"Predicted MTTF for {model_name} ({prediction_label_aft}): May not be defined or directly computable for certain LogLogistic parameters.")

                                all_results_summary.append({"Model": model_name, "AIC": f"{aic_val:.2f}", "Log-Likelihood": f"{log_lik:.2f}", "Converged": converged_status_aft, "Median Survival": median_aft_val_str, "MTTF": mttf_aft_val_str})

                            except ConvergenceError:
                                st.error(f"The AFT model ({model_name}) did not converge.")
                                explain_convergence_error()
                                all_results_summary.append({"Model": model_name, "AIC": "Convergence Error", "Log-Likelihood": "Convergence Error", "Converged": False, "Median Survival": "N/A", "MTTF": "N/A"})
                                print(f"Traceback for AFT ConvergenceError ({model_name}):")
                                print(traceback.format_exc())
                    except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
                        st.error(f"Could not fit {model_name}. Error: {e}")
                        print(f"Traceback for {model_name} fitting error:"); print(traceback.format_exc())
                        all_results_summary.append({"Model": model_name, "AIC": "Error", "Log-Likelihood": "Error", "Converged": "Error", "Median Survival": "Error", "MTTF": "Error"})

                    except Exception as e:
                        st.error(f"An unexpected error occurred while fitting {model_name}: {e}")
                        print(f"Traceback for unexpected error in {model_name}:"); print(traceback.format_exc())
                        all_results_summary.append({"Model": model_name, "AIC": "Error", "Log-Likelihood": "Error", "Converged": "Error", "Median Survival": "Error", "MTTF": "Error"})


                st.write("## Summary of All Fitted Models")
                if all_results_summary:
                    for res_dict in all_results_summary:
                        res_dict.setdefault("Median Survival", "N/A")
                        res_dict.setdefault("MTTF", "N/A")

                    results_df = pd.DataFrame(all_results_summary)
                    best_aic_val = np.inf
                    best_ll_val = -np.inf
                    best_model_name_summary_aic = "N/A"
                    best_model_name_summary_ll = "N/A"
                    valid_results_for_best = []

                    for r_idx, r_row in results_df.iterrows():
                        converged_val = r_row.get("Converged")
                        is_converged = False
                        if isinstance(converged_val, bool): is_converged = converged_val
                        elif isinstance(converged_val, str) and converged_val.lower() not in ["n/a", "error", "data insufficient", "convergence error"]:
                            try: is_converged = bool(converged_val)
                            except: pass

                        if is_converged:
                            try:
                                aic = float(r_row.get("AIC", np.nan))
                                ll = float(r_row.get("Log-Likelihood", np.nan))
                                if not np.isnan(aic):
                                    valid_results_for_best.append({
                                        "Model": r_row["Model"],
                                        "AIC_float": aic,
                                        "LL_float": ll if not np.isnan(ll) else -np.inf
                                    })
                            except: continue

                    results_df["Is Best (AIC)"] = ""
                    results_df["Is Best (LogLik)"] = ""

                    if valid_results_for_best:
                        sorted_by_aic = sorted(valid_results_for_best, key=lambda x: x["AIC_float"])
                        if sorted_by_aic:
                            best_model_name_summary_aic = sorted_by_aic[0]["Model"]
                            best_aic_val = sorted_by_aic[0]["AIC_float"]
                            results_df.loc[results_df["Model"] == best_model_name_summary_aic, "Is Best (AIC)"] = "🏆"

                        valid_ll_results = [r for r in valid_results_for_best if not np.isinf(r["LL_float"])]
                        if valid_ll_results:
                            sorted_by_ll = sorted(valid_ll_results, key=lambda x: x["LL_float"], reverse=True)
                            if sorted_by_ll:
                                best_model_name_summary_ll = sorted_by_ll[0]["Model"]
                                best_ll_val = sorted_by_ll[0]["LL_float"]
                                results_df.loc[results_df["Model"] == best_model_name_summary_ll, "Is Best (LogLik)"] = "🌟"

                    st.dataframe(results_df)

                    if best_model_name_summary_aic != "N/A":
                        st.info(f"🏆 **Best model by AIC:** {best_model_name_summary_aic} (AIC: {best_aic_val:.2f})")
                    if best_model_name_summary_ll != "N/A" and best_model_name_summary_ll != best_model_name_summary_aic:
                        st.info(f"🌟 **Best model by Log-Likelihood:** {best_model_name_summary_ll} (Log-Likelihood: {best_ll_val:.2f})")
                    elif best_model_name_summary_aic == "N/A" and best_model_name_summary_ll == "N/A":
                        st.warning("No models converged successfully with valid AIC/Log-Likelihood values to determine a best model.")
                else:
                    st.info("No models were selected or processed.")
