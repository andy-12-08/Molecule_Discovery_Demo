import streamlit as st
import pandas as pd
from src.pretrainedmodel import GNNmodel
from rdkit import Chem

graph_model_path = 'models'

st.title('Predictive Models')
st.write('Predictive models are used to predict the properties of molecules based on their structure. These models can be broadly classified into two categories:')

tab1, tab2 = st.tabs(['Descriptor based predictors', 'Graph based predictors'])

with tab1:
    st.write('Predictive models that use molecular descriptors for property prediction.')

with tab2:
    st.write('Models that leverage molecular graphs for advanced property prediction.')
    with st.expander('🔵 Predict_SurfMOL03212024'):  
        smiles_input = st.text_input('Enter a molecule SMILES')
        predict_button = st.button('Predict')
        if predict_button:
            prediction_model = GNNmodel(path = graph_model_path)
            smiles_list = [smiles.strip() for smiles in smiles_input.split(',') if smiles.strip()]

            # check for invalid SMILES using rdkit and remove them
            invalid_smiles = []
            for smiles in smiles_list:
                if not Chem.MolFromSmiles(smiles):
                    invalid_smiles.append(smiles)
            if invalid_smiles:
                st.write(f'Invalid SMILES: {", ".join(invalid_smiles)}')
                smiles_list = [smiles for smiles in smiles_list if smiles not in invalid_smiles]
        
            property_values = []
            for i in range(len(smiles_list)):
                pred_value = prediction_model.predict([smiles_list[i]])
                property_values.append(pred_value[0])
            property_values = [round(value, 2) for value in property_values]
            prediction_df = pd.DataFrame({'SMILES': smiles_list, 'Predicted Property Value': property_values})
            st.write(prediction_df)