import streamlit as st
from PIL import Image

import functions
import show_parametricmodel , show_parametricmix, show_otherfunc, \
    show_repairable, show_alt, show_fitter, show_rdt, show_sndiagram, \
    show_palmgrenminer, show_fracturemechanics, show_creep, \
    show_accelerationfactor


from __init__ import __version__
import authentication_streamlit


authentication_streamlit.check_authentication(
    "Denied access. Please log in to https://ceerma.org"
)


image_ufpe = Image.open('./src/logo.png')
image_pip = Image.open('./src/logopip.png')
image_ceerma = Image.open('./src/favicon.png')

st.set_page_config(page_title="ReMDA",
                   page_icon=image_ceerma,layout="wide",
                   initial_sidebar_state="expanded")

version_info = f"Version {__version__}"
st.sidebar.markdown(
        f"""
        <div style="display:table;margin-top:-80px">{version_info}</div>
        """,
        unsafe_allow_html=True,
)

st.sidebar.image(image_ufpe)

st.sidebar.title("📈 ReMDA")

st.sidebar.caption("Reliability Modelling and Data Analysis")

with st.sidebar:
    functions.write_text_align(
        """
        This app is an easy-to-use interface built in Streamlit for reliability-related analysis and visualization using the 
        <a href="https://reliability.readthedocs.io/en/latest/index.html" target="_blank">Reliability</a> and 
        <a href="https://lifelines.readthedocs.io/en/latest/index.html" target="_blank">Lifelines</a> Python libraries.
        """,
        align='justify'
    )


submodules_parametric = {
    "Select a submodule": lambda: None,
    "Probability Distributions": show_parametricmodel.show,
    "Mixture Models": show_parametricmix.show,
    # "Non-Parametric Model": show_comingsoon.show,
    # "Fit Distribution": show_fitter.show,
}

submodules_physics = {
    "Select a submodule": lambda: None,
    "S-N diagram": show_sndiagram.show,
    "Fracture Mechanics": show_fracturemechanics.show,
    "Creep": show_creep.show,
    "Palmgren-Miner": show_palmgrenminer.show,
    "Acceleration Factor": show_accelerationfactor.show,
}

modules = {
    "Select a module": lambda: None,
    "Parametric Models": submodules_parametric,
    "Fit Distribution": show_fitter.show,
    "Accelerated Life Testing": show_alt.show,
    "Reliability Demonstration Tests": show_rdt.show,
    "Repairable Systems": show_repairable.show,
    # "Other Functions"
    "Stress and Strength": show_otherfunc.show,
    "Physics of Failure": submodules_physics,
}

menu = st.sidebar.selectbox(" ", list(modules), label_visibility="collapsed")

if isinstance(modules[menu], dict):
    # Se o módulo tiver submódulos
    submenu = st.sidebar.selectbox(" ", list(modules[menu]), label_visibility="collapsed")
    if submenu == "Select a submodule":
        functions.page_config(hide_menu=True)
    else:
        functions.page_config(submenu, hide_menu=True)
        modules[menu][submenu]()
else:
    if menu == "Select a module":
        functions.page_config(hide_menu=True)
    else:
        functions.page_config(menu, hide_menu=True)
        modules[menu]()