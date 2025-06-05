import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from reliability.Distributions import Weibull_Distribution, Lognormal_Distribution, Exponential_Distribution, Normal_Distribution
from reliability.Probability_plotting import Weibull_probability_plot, Lognormal_probability_plot, Exponential_probability_plot, Normal_probability_plot
from reliability.Fitters import Fit_Weibull_2P, Fit_Lognormal_2P, Fit_Exponential_1P, Fit_Normal_2P
from scipy.stats import linregress

# --- Life-Stress Models (Permanece o mesmo) ---
LIFE_STRESS_MODELS = {
    "Power": { # L = A * S^B
        "func": lambda S, A, B: A * S**B if A > 0 and np.all(S > 0) else np.nan, # S > 0 para Power
        "transform_S": lambda s: np.log(s) if np.all(s > 0) else np.full_like(s, np.nan, dtype=float),
        "transform_L": np.log,
        "A_calc": lambda intercept, slope: np.exp(intercept), "B_calc": lambda intercept, slope: slope,
        "param_A_desc": "A (coefficient)", "param_B_desc": "B (exponent)"
    },
    "Exponential": { # L = A * exp(B*S)
        "func": lambda S, A, B: A * np.exp(B * S) if A > 0 else np.nan,
        "transform_S": lambda x: x, "transform_L": np.log,
        "A_calc": lambda intercept, slope: np.exp(intercept), "B_calc": lambda intercept, slope: slope,
        "param_A_desc": "A (coefficient)", "param_B_desc": "B (rate in exponent)"
    },
    "Arrhenius": { # L = A * exp(B/S)
        "func": lambda S, A, B: A * np.exp(B / S) if A > 0 and np.all(S != 0) else np.nan, # S != 0 para Arrhenius
        "transform_S": lambda s: 1/s if np.all(s != 0) else np.full_like(s, np.nan, dtype=float),
        "transform_L": np.log,
        "A_calc": lambda intercept, slope: np.exp(intercept), "B_calc": lambda intercept, slope: slope,
        "param_A_desc": "A (coefficient)", "param_B_desc": "B (activation energy related)"
    },
    "Linear": { # L = A*S + B
        "func": lambda S, A_slope, B_intercept: A_slope * S + B_intercept,
        "transform_S": lambda x: x, "transform_L": lambda x: x,
        "A_calc": lambda intercept, slope: slope, "B_calc": lambda intercept, slope: intercept,
        "param_A_desc": "A (slope)", "param_B_desc": "B (intercept)"
    },
    "Logarithmic": { # L = A + B*ln(S)
        "func": lambda S, A, B: A + B * np.log(S) if np.all(S > 0) else np.nan, # S > 0 para Logarithmic
        "transform_S": lambda s: np.log(s) if np.all(s > 0) else np.full_like(s, np.nan, dtype=float),
        "transform_L": lambda x: x,
        "A_calc": lambda intercept, slope: intercept, "B_calc": lambda intercept, slope: slope,
        "param_A_desc": "A (intercept)", "param_B_desc": "B (coefficient of ln(S))"
    }
}

# --- Funções Auxiliares de Excel (Permanece o mesmo) ---
def create_example_excel_df():
    data = {
        'Time': np.concatenate([
            Weibull_Distribution(alpha=2000, beta=2.5).random_samples(12, seed=1),
            Weibull_Distribution(alpha=1000, beta=2.5).random_samples(12, seed=2),
            Weibull_Distribution(alpha=400,  beta=2.5).random_samples(12, seed=3)
        ]),
        'Stress1': np.concatenate([np.full(12, 50), np.full(12, 75), np.full(12, 100)])
    }
    return pd.DataFrame(data)

def get_excel_download_link(df_example):
    output = BytesIO()
    with pd.ExcelWriter(output) as writer:
        df_example.to_excel(writer, index=False, sheet_name='ALT_Data')
    return output.getvalue()

# --- Novas Funções de Análise Modularizadas ---

def _generate_prob_plot_for_single_dist(df_main, dist_name, display_list):
    """
    Gera gráfico de probabilidade e estima parâmetros para uma única distribuição.
    Adiciona mensagens de log à display_list.
    Retorna (figure_object, parameters_dataframe) ou (None, None) em caso de falha.
    """
    unique_stresses = sorted(df_main['Stress1'].unique())
    prop_cycler = plt.rcParams['axes.prop_cycle']
    default_colors = prop_cycler.by_key()['color']
    
    fig, ax = plt.subplots(figsize=(7, 5))
    current_dist_data_for_df = []
    plot_successful_for_dist = False

    for i, stress_level_val in enumerate(unique_stresses):
        stress_specific_data = df_main[df_main['Stress1'] == stress_level_val]['Time'].dropna().values
        if len(stress_specific_data) < 2:
            display_list.append({'type': 'caption', 'content': f"Stress {stress_level_val} for {dist_name}: Insufficient data ({len(stress_specific_data)} pts), skipping."})
            continue

        current_plot_color = default_colors[i % len(default_colors)]
        char_life_val, r_squared_val, params_info_str = np.nan, np.nan, "Fit Error"
        plot_label = f'S={stress_level_val:.2g}'

        try:
            if dist_name == "Weibull":
                fit_params = Fit_Weibull_2P(failures=stress_specific_data, print_results=False, show_probability_plot=False)
                if fit_params.alpha is not None:
                    char_life_val, params_info_str = fit_params.alpha, f"β={fit_params.beta:.3f}, η={fit_params.alpha:.3f}"
                Weibull_probability_plot(failures=stress_specific_data, label=plot_label, color=current_plot_color)
            elif dist_name == "Lognormal":
                fit_params = Fit_Lognormal_2P(failures=stress_specific_data, print_results=False, show_probability_plot=False)
                if fit_params.mu is not None:
                    char_life_val = np.exp(fit_params.mu)
                    params_info_str = f"μ={fit_params.mu:.3f}, σ={fit_params.sigma:.3f}, Med={char_life_val:.3f}"
                Lognormal_probability_plot(failures=stress_specific_data, label=plot_label, color=current_plot_color)
            elif dist_name == "Exponential":
                fit_params = Fit_Exponential_1P(failures=stress_specific_data, print_results=False, show_probability_plot=False)
                if fit_params.Lambda is not None:
                    char_life_val = 1 / fit_params.Lambda if fit_params.Lambda != 0 else np.nan
                    params_info_str = f"λ={fit_params.Lambda:.3E}, Mean={char_life_val:.3f}"
                Exponential_probability_plot(failures=stress_specific_data, label=plot_label, color=current_plot_color)
            elif dist_name == "Normal":
                fit_params = Fit_Normal_2P(failures=stress_specific_data, print_results=False, show_probability_plot=False)
                if fit_params.mu is not None:
                    char_life_val = fit_params.mu
                    params_info_str = f"μ={fit_params.mu:.3f}, σ={fit_params.sigma:.3f}"
                Normal_probability_plot(failures=stress_specific_data, label=plot_label, color=current_plot_color)
            
            if not np.isnan(char_life_val):
                current_dist_data_for_df.append({
                    "Stress": stress_level_val, "Characteristic_Life": char_life_val,
                    "Est_Parameters_at_Stress": params_info_str
                })
                plot_successful_for_dist = True
        except Exception as e:
            display_list.append({'type': 'caption', 'content': f"Error fitting/plotting {dist_name} at S={stress_level_val}: {str(e)[:100]}"})

    if plot_successful_for_dist:
        ax.legend(fontsize='small', title=f"{dist_name} Parameters by Stress")
        ax.tick_params(axis='both', which='major', labelsize=8)
        param_df = pd.DataFrame(current_dist_data_for_df)
        return fig, param_df
    else:
        plt.close(fig) # Fecha a figura se nada foi plotado nela
        return None, pd.DataFrame(current_dist_data_for_df) # Retorna DF mesmo que vazio, para checagens

def _fit_ls_model_and_extrapolate_single_dist(df_char_life_for_model, ls_model_name, use_stress_val, dist_name, display_list):
    """
    Ajusta modelo vida-estresse e extrapola. Adiciona resultados à display_list.
    """
    df_char_life_for_model.dropna(subset=['Stress', 'Characteristic_Life'], inplace=True)
    if len(df_char_life_for_model) < 2:
        display_list.append({'type': 'error', 'content': f"Insufficient valid data points ({len(df_char_life_for_model)}) from {dist_name} prob. plot to fit {ls_model_name} model. Need at least 2."})
        return

    model_details = LIFE_STRESS_MODELS[ls_model_name]
    S_original = df_char_life_for_model['Stress'].values
    L_original = df_char_life_for_model['Characteristic_Life'].values

    try:
        S_transformed = model_details["transform_S"](S_original)
        L_transformed = model_details["transform_L"](L_original)
    except Exception as e_transform:
        display_list.append({'type': 'error', 'content': f"Error during data transformation for {ls_model_name} model: {e_transform}"})
        return
        
    valid_idx = ~np.isnan(S_transformed) & ~np.isinf(S_transformed) & \
                ~np.isnan(L_transformed) & ~np.isinf(L_transformed)
    
    S_tfm_clean, L_tfm_clean = S_transformed[valid_idx], L_transformed[valid_idx]
    S_orig_clean, L_orig_clean = S_original[valid_idx], L_original[valid_idx]

    if len(S_tfm_clean) < 2:
        display_list.append({'type': 'error', 'content': f"After data transformation for {ls_model_name}, < 2 valid data points remain for {dist_name}."})
        return

    try:
        slope, intercept, r_val, _, _ = linregress(S_tfm_clean, L_tfm_clean)
    except ValueError as ve:
        display_list.append({'type': 'error', 'content': f"Linear regression failed for {ls_model_name} on {dist_name} data. Error: {ve}"})
        return
        
    r_squared_ls_model = r_val**2
    param_A_ls = model_details["A_calc"](intercept, slope)
    param_B_ls = model_details["B_calc"](intercept, slope)

    display_list.append({'type': 'markdown', 'content': f"**Life-Stress Model Details ({ls_model_name}):**"})
    display_list.append({'type': 'markdown', 'content': f"* {model_details['param_A_desc']}: `{param_A_ls:,.4g}`"})
    display_list.append({'type': 'markdown', 'content': f"* {model_details['param_B_desc']}: `{param_B_ls:,.4f}`"})
    display_list.append({'type': 'markdown', 'content': f"* R-squared (of transformed linear fit): `{r_squared_ls_model:.4f}`"})

    fig_ls, ax_ls = plt.subplots(figsize=(8,6))
    ax_ls.scatter(S_orig_clean, L_orig_clean, color='dodgerblue', label='Observed Char. Lives', zorder=5)
    
    s_plot_min = min(S_orig_clean.min() if S_orig_clean.size > 0 else use_stress_val, use_stress_val) * 0.8
    s_plot_max = max(S_orig_clean.max() if S_orig_clean.size > 0 else use_stress_val, use_stress_val) * 1.2
    s_plot_min_safe = max(s_plot_min, 1e-9) if s_plot_min <=0 else s_plot_min
    
    log_scale_stress_axis = (S_orig_clean.size > 0 and S_orig_clean.max() / S_orig_clean.min() > 10) or \
                             ls_model_name in ["Power", "Arrhenius", "Logarithmic"]

    if log_scale_stress_axis and s_plot_min_safe > 0 and s_plot_max > 0 :
        s_plot_points = np.geomspace(s_plot_min_safe, s_plot_max, 200)
    else:
        s_plot_points = np.linspace(s_plot_min_safe, s_plot_max, 200)
    s_plot_points = np.unique(s_plot_points)

    try:
        l_plot_fitted_curve = model_details["func"](s_plot_points, param_A_ls, param_B_ls)
        ax_ls.plot(s_plot_points, l_plot_fitted_curve, color='red', linestyle='--', label=f'Fitted {ls_model_name} Model')
    except Exception as e_plot_ls:
        display_list.append({'type': 'warning', 'content': f"Could not plot the fitted life-stress curve: {e_plot_ls}"})

    ax_ls.set_xlabel("Stress Level")
    ax_ls.set_ylabel(f"Characteristic Life (from {dist_name})")
    ax_ls.set_title(f"Life-Stress Plot: {ls_model_name} Model for {dist_name} Data")
    
    if L_orig_clean.size > 0 and not np.all(L_orig_clean <= 0) and (L_orig_clean.max() / L_orig_clean.min() > 10):
         ax_ls.set_yscale('log')
    if log_scale_stress_axis : 
        ax_ls.set_xscale('log')
    
    ax_ls.legend()
    display_list.append({'type': 'pyplot', 'content': fig_ls})

    # Extrapolation
    display_list.append({'type': 'markdown', 'content': "### 3. Extrapolation to Use Condition"})
    try:
        extrapolated_life_val = model_details["func"](use_stress_val, param_A_ls, param_B_ls)
        if np.isnan(extrapolated_life_val) or np.isinf(extrapolated_life_val):
            display_list.append({'type': 'error', 'content': f"Extrapolated life at use stress ({use_stress_val:.3g}) resulted in an invalid value (NaN/Inf). Check model, parameters, or use stress."})
        else:
            display_list.append({'type': 'success', 'content': f"**Predicted Life at Use Stress ({use_stress_val:.3g}) based on {dist_name} & {ls_model_name} model: **{extrapolated_life_val:,.3g}** time units**"})

    except Exception as e_extrap:
        display_list.append({'type': 'error', 'content': f"Could not extrapolate life or determine parameters at use stress: {e_extrap}"})


def run_full_analysis_and_prepare_display(df_main, distributions_to_analyze, ls_model_name, use_stress):
    """
    Orquestra a análise completa para cada distribuição e prepara os resultados para exibição.
    """
    results_list_for_display = [] 

    for dist_name in distributions_to_analyze:
        results_list_for_display.append({'type': 'markdown', 'content': f"<hr><h2>Analysis for {dist_name} Distribution</h2>", 'unsafe_allow_html': True})

        # 1. Probability Plotting and Parameter Estimation
        results_list_for_display.append({'type': 'subheader', 'content': f"1. {dist_name} Probability Plot & Estimated Parameters"})
        
        fig_prob, df_params_at_stress = _generate_prob_plot_for_single_dist(df_main, dist_name, results_list_for_display)
        
        if fig_prob:
            results_list_for_display.append({'type': 'pyplot', 'content': fig_prob})
        
        if df_params_at_stress is not None and not df_params_at_stress.empty:
            results_list_for_display.append({'type': 'dataframe', 'content': df_params_at_stress, 'height': 180}) # Aumentar altura
            if df_params_at_stress['Characteristic_Life'].isnull().all():
                results_list_for_display.append({'type': 'warning', 'content': f"Could not determine characteristic life for {dist_name} from probability plot. Skipping Life-Stress modeling for this distribution."})
                continue # Pula para a próxima distribuição se não houver vida característica
        else: 
            results_list_for_display.append({'type': 'warning', 'content': f"Failed to extract parameters or generate probability plot for {dist_name}. Skipping Life-Stress modeling."})
            continue # Pula para a próxima distribuição

        # 2. Life-Stress Model Fitting & Extrapolation
        results_list_for_display.append({'type': 'subheader', 'content': f"2. Life-Stress Model ({ls_model_name}) & Extrapolation for {dist_name}"})
        _fit_ls_model_and_extrapolate_single_dist(df_params_at_stress, ls_model_name, use_stress, dist_name, results_list_for_display)
        
    st.session_state.analysis_results_display = results_list_for_display


def show():
    st.title("ALT model estimation")

    # --- Guias e Informações (mantidos) ---
    st.markdown("This module performs ALT data analysis for each selected distribution using a chosen life-stress model.")
    with st.expander("Short Guide & Methodology"):
        st.markdown("""
        **Methodology Overview:**
        - **Upload Data:** Provide an Excel file with failure times and stress levels.
        - **Configure Analysis:** Select one or more statistical distributions, a life-stress model, and input the normal operating stress.
        - **Run Analysis:** Click Fit Model(s)' button.
        - **Review Results:** For each selected distribution, the tool will:
            1.  Generate a probability plot and estimate distribution parameters at each stress level.
            2.  Fit the chosen life-stress model to the characteristic life data obtained.
            3.  Extrapolate to estimate life at the normal operating stress.
            4.  Display all relevant plots, tables, and extrapolated values.
        """)
    with st.expander("Data Format (Excel .xlsx)"):
        st.markdown("Your Excel file should contain `Time` and `Stress1` columns.")
        df_example = create_example_excel_df()
        st.dataframe(df_example.head())
        st.download_button(label="Download Example Excel File", data=get_excel_download_link(df_example),
                           file_name="alt_single_button_example.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # --- Upload de Dados ---
    uploaded_file = st.file_uploader("Upload ALT data (.xlsx format)", type="xlsx", key="file_uploader")
    
    if 'df_processed' not in st.session_state: st.session_state.df_processed = None
    if 'analysis_results_display' not in st.session_state: st.session_state.analysis_results_display = []
    if 'last_analysis_inputs' not in st.session_state: st.session_state.last_analysis_inputs = {}


    if uploaded_file:
        try:
            df_input = pd.read_excel(uploaded_file)
            st.success("File uploaded successfully!")
            if "Time" not in df_input.columns or "Stress1" not in df_input.columns:
                st.error("Data must include 'Time' and 'Stress1' columns.")
                st.session_state.df_processed = None
                st.session_state.analysis_results_display = [] 
            else:
                df_input["Time"] = pd.to_numeric(df_input["Time"], errors='coerce')
                df_input["Stress1"] = pd.to_numeric(df_input["Stress1"], errors='coerce')

                if df_input["Time"].isnull().any() or df_input["Stress1"].isnull().any():
                    st.error("Non-numeric values detected in 'Time' or 'Stress1'. Please check data.")
                    st.session_state.df_processed = None
                    st.session_state.analysis_results_display = []
                elif (df_input["Time"] <= 0).any():
                    st.warning("Warning: Non-positive 'Time' values found. This may affect log transformations or certain distributions.")
                    st.session_state.df_processed = df_input
                    st.session_state.analysis_results_display = []
                else:
                    st.session_state.df_processed = df_input
                    st.session_state.analysis_results_display = [] 
                
                if st.session_state.df_processed is not None:
                    st.subheader("Uploaded Data Preview (First 5 Rows):")
                    st.dataframe(st.session_state.df_processed.head())

        except Exception as e:
            st.error(f"Error reading/processing file: {e}")
            st.session_state.df_processed = None
            st.session_state.analysis_results_display = []
    
    if st.session_state.df_processed is not None:
        st.header("Analysis Configuration")
        
        # Restaurar ou usar default para seleções
        last_inputs = st.session_state.last_analysis_inputs
        available_distributions = ["Weibull", "Lognormal", "Exponential", "Normal"]
        default_dist_sel = last_inputs.get('selected_distributions', available_distributions[:1])
        
        life_stress_model_names = list(LIFE_STRESS_MODELS.keys())
        default_ls_sel = last_inputs.get('selected_ls_model', life_stress_model_names[0])
        default_ls_idx = life_stress_model_names.index(default_ls_sel) if default_ls_sel in life_stress_model_names else 0
        
        min_data_stress = st.session_state.df_processed['Stress1'].min() if 'Stress1' in st.session_state.df_processed else 1.0
        default_use_stress_val = max(0.001, min_data_stress / 2 if pd.notnull(min_data_stress) and min_data_stress > 0 else 1.0)
        default_use_stress_val = last_inputs.get('use_stress', default_use_stress_val)

        # Widgets de configuração
        cols_config = st.columns(3)
        with cols_config[0]:
            selected_distributions = st.multiselect(
                "1. Select Distributions for Analysis:", available_distributions, default=default_dist_sel, key="dist_multi_select"
            )
        with cols_config[1]:
            selected_ls_model = st.selectbox(
                "2. Select Life-Stress Model:", life_stress_model_names, index=default_ls_idx, key="ls_model_select_single"
            )
        with cols_config[2]:
            use_stress = st.number_input(
                "3. Normal Operating Stress:", value=default_use_stress_val, format="%.4f", min_value=0.000001, key="use_stress_single"
            )

        if st.button("Fit Model(s)", type="primary", key="run_full_analysis_button"):
            if not selected_distributions:
                st.error("Please select at least one distribution to analyze.")
            else:
                # Armazena as entradas atuais para persistência entre reexecuções se não houver nova análise
                st.session_state.last_analysis_inputs = {
                    'selected_distributions': selected_distributions,
                    'selected_ls_model': selected_ls_model,
                    'use_stress': use_stress
                }
                # Limpar resultados anteriores antes de uma nova análise
                st.session_state.analysis_results_display = [] 
                with st.spinner("Performing analysis... This may take a moment."):
                    run_full_analysis_and_prepare_display(
                        st.session_state.df_processed,
                        selected_distributions,
                        selected_ls_model,
                        use_stress
                    )
        
        # Exibir resultados armazenados
        if st.session_state.analysis_results_display:
            for item in st.session_state.analysis_results_display:
                item_type = item.get('type')
                content = item.get('content')
                if item_type == 'markdown':
                    st.markdown(content, unsafe_allow_html=item.get('unsafe_allow_html', False))
                elif item_type == 'subheader':
                    st.subheader(content)
                elif item_type == 'dataframe':
                    st.dataframe(content, height=item.get('height'))
                elif item_type == 'pyplot':
                    if content is not None:
                        st.pyplot(content)
                        plt.close(content) 
                elif item_type == 'warning':
                    st.warning(content)
                elif item_type == 'error':
                    st.error(content)
                elif item_type == 'success':
                    st.success(content)
                elif item_type == 'info':
                    st.info(content)
                elif item_type == 'caption':
                    st.caption(content)

    elif uploaded_file is None and not st.session_state.df_processed :
         st.info("Please upload an Excel data file to begin the analysis.")