import streamlit as st

import authenticator


def check_authentication(
    message="Acesso não autorizado. Favor fazer login em https://ceerma.org"
    ):
    authorized = False

    # Older streamlit versions (before around 1.30), use the 4 lines below:
    # params = st.experimental_get_query_params()
    # token = params.get("token")
    # if token != None:
    #     token = token[0]

    # Newer streamlit versions (after around 1.30), use the 2 lines below:
    params = st.query_params
    token = params.get("token")


    if token != None:
        token_data = authenticator.validate_token(token)
        if token_data != None:
            authorized = True

    if authorized != True:
        st.error(message)
        st.stop()


#### Usage: paste the following code to your app.py (or similar) streamlit file
# import authentication_streamlit
# authentication_streamlit.check_authentication()
