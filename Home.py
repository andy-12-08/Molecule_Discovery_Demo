import numpy as np
import streamlit as st

st.set_page_config(page_title="Molecule Design and Discovery", 
                   page_icon="🧊",
                   layout="wide", 
                   initial_sidebar_state="expanded",
                   )

#st.title("SingletonAI")
st.title("Welcome to Romagnolis' Generative AI Platform")
#st.write("Welcome to SingletonAI, a platform for molecule design and discovery using generative AI. This platform is designed to help you generate novel molecules with desired properties. You can use the tools provided here to generate molecules, visualize them, and analyze their properties. You can also use the platform to train your own models and generate molecules using your own data.")

# st.markdown("<h5 style='color:blue'> Developed and Deployed by Andrew Okafor <h5>", unsafe_allow_html=True)
# add a header
st.header('Demonstration of Molecule Discovery Using Generative AI')

st.image('images/molecules.gif')
