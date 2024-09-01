import numpy as np
import streamlit as st

st.set_page_config(page_title="Molecule Design and Discovery", 
                   page_icon="🧊",
                   layout="wide", 
                   initial_sidebar_state="expanded",
                   )

st.title("Welcome to Romagnoli's Group")
# add a header
st.header('Demonstration of Molecule Discovery Using Generative AI')

st.image('images/molecules.gif')
