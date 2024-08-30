import streamlit as st

st.title('Predictive Models')
st.write('Predictive models are used to predict the properties of molecules based on their structure. These models can be broadly classified into two categories:')

tab1, tab2 = st.tabs(['Descriptor based predictors', 'Graph based predictors'])

with tab1:
    st.write('Predictive models that use molecular descriptors for property prediction.')

with tab2:
    st.write('Models that leverage molecular graphs for advanced property prediction.')