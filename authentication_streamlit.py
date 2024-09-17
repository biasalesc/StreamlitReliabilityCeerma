import streamlit as st

import authenticator


def check_authentication(
    message="Acesso não autorizado. Favor fazer login em https://ceerma.org"
    ):
    authorized = False

    params = st.experimental_get_query_params()
    token = params.get("token")
    if token != None:
        token = token[0]

    if token != None:
        token_data = authenticator.validate_token(token)
        if token_data != None:
            authorized = True

    if authorized != True:
        st.error(message)
        st.stop()
