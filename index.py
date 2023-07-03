import streamlit as st
from PIL import Image

import functions
import show_parametricmodel , show_parametricmix, show_otherfunc, \
    show_repairable, show_alt, show_fitter, show_rdt

from __init__ import __version__


image_ufpe = Image.open('./src/logo.png')
image_pip = Image.open('./src/logopip.png')
image_ceerma = Image.open('./src/favicon.png')

st.set_page_config(page_title="Reliability",
                   page_icon=image_ceerma,layout="wide",
                   initial_sidebar_state="expanded")

version_info = f"Version {__version__}"
st.sidebar.markdown(
        f"""
        <div style="display:table;margin-top:-32%;margin-left:0%;">{version_info}</div>
        """,
        unsafe_allow_html=True,
)

st.sidebar.image(image_ufpe)

st.sidebar.title("📈 Reliability app")

st.sidebar.write("""
This app is an easy-to-use interface built in Streamlit for reliability
related analysis and visualization using the Reliability Python library.
""")

submodules = {
    "Select a submodule": lambda: None,
    "Probability Distributions": show_parametricmodel.show,
    "Mixture Models": show_parametricmix.show,
    # "Non-Parametric Model": show_comingsoon.show,
    # "Fit Distribution": show_fitter.show,
}

modules = {
    "Select a module": lambda: None,
    "Parametric Models": submodules,
    "Fit Distribution": show_fitter.show,
    "Accelerated Life Testing": show_alt.show,
    "Repairable Systems": show_repairable.show,
    # "Other Functions"
    "Stress and Strength": show_otherfunc.show,
    "RDT": show_rdt.show,
}

menu = st.sidebar.selectbox(" ", list(modules), label_visibility="collapsed")

if menu == list(modules)[1]:
    submenu = st.sidebar.selectbox(" ", list(modules[menu]), label_visibility="collapsed")
    if submenu == "Select a submodule":
        functions.page_config()
    else:
        functions.page_config(submenu)
    modules[menu][submenu]()
else:
    if menu == "Select a module":
        functions.page_config()
    else:
        functions.page_config(menu)
    modules[menu]()
