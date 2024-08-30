import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import random
import selfies as sf
import torch
import sys
import os
sys.path.append('../')
from src.vae_module import VAE
from src.vae_module import flatten

# load the trained VAE model
latent_dim = 32  # latent dimension of the VAE
vae = VAE(input_dim=3624, latent_dim=latent_dim)
vae.load_state_dict(torch.load('models/vae_model.pt'))
original_data = pd.read_csv('data/test_more_dataNoinionc.csv', header=None, names=['smiles', 'logCMC'])
original_data_input_smiles_list = original_data.iloc[:, 0].tolist()
    
def encode_data_from_smiles_to_vae_encoded(input_smiles_list):
    input_selfies_list = list(map(sf.encoder, input_smiles_list))
    # Parameters for encoding
    max_len = max(sf.len_selfies(s) for s in input_selfies_list)
    alphabet = sf.get_alphabet_from_selfies(input_selfies_list)
    alphabet.add("[nop]")
    alphabet = list(sorted(alphabet))
    vocab_stoi = {symbol: idx for idx, symbol in enumerate(alphabet)}
    vocab_itos = {idx: symbol for symbol, idx in vocab_stoi.items()}
    # Convert SELFIES to one-hot encoding
    input_one_hot_arr = np.array([sf.selfies_to_encoding(s, vocab_stoi, pad_to_len=max_len)[1] for s in input_selfies_list])
    ## Convert the "input_one_hot_arr" dataset to tensor
    input_one_hot_arr_tensor = torch.tensor(input_one_hot_arr, dtype=torch.float32)
    # Flatten the input data using the custom 'flatten' function
    # This function takes a 3D tensor 'x_train' and reshapes it into a 2D tensor
    width, height, input_dim, flattened_dataset = flatten(input_one_hot_arr_tensor)
    with torch.no_grad():
        # Pass the generated molecules through the VAE encoder
        encoded_flattened_dataset = vae.encoder(flattened_dataset)
    return encoded_flattened_dataset, height, vocab_itos, width


def generate_vae_molecules(n, original_data_input_smiles_list, height, vocab_itos, width):
    # Generate molecules from the trained VAE
    vae.eval()
    num_samples = 10*n  # generate enough samples to select n molecules
    with torch.no_grad():
        latent_samples = torch.randn(num_samples, latent_dim)
        generated_molecules = vae.decoder(latent_samples)
        # Convert the generated molecules to SMILES
        def generated_molecules_to_smiles(generated_molecules):
            # Reshape satisfying_molecules_tensor back to a 3D tensor
            generated_molecules_tensor_3d = generated_molecules.view(-1, width, height)
            # Convert the PyTorch 3D tensor to a NumPy array
            generated_molecules_numpy = generated_molecules_tensor_3d.numpy()
            max_values = np.max(generated_molecules_numpy, axis=2, keepdims=True)
            generated_data = np.where(generated_molecules_numpy == max_values, 1, 0)
            ### Reproduce SMILES list and visualize the output images
            output_smiles_list = []
            for i in range (0,len(generated_data)):
                sm = sf.decoder(sf.encoding_to_selfies(generated_data[i].tolist(), vocab_itos, enc_type="one_hot"))
                output_smiles_list.append(sm)
            return output_smiles_list
        # Convert the generated molecules to SMILES
        generated_molecules_smiles_list = generated_molecules_to_smiles(generated_molecules)
        # remove duplicates
        generated_molecules_smiles_list = list(set(generated_molecules_smiles_list))
        # remove molecules that are in the original dataset
        generated_molecules_smiles_list = [m for m in generated_molecules_smiles_list if m not in original_data_input_smiles_list]
        # select n molecules randomly from the generated molecules
        selected_n_molecules = random.sample(generated_molecules_smiles_list, n)
        return selected_n_molecules


        
def features_df(smiles_list):
    # Get a list of descriptor functions
    descriptor_list = [desc[0] for desc in Descriptors._descList]
    descriptor_list = descriptor_list[:100]  # Limit to first 100 descriptors for simplicity
    # drop ['NumRadicalElectrons'] descriptors
    descriptor_list = [desc for desc in descriptor_list if desc not in ['NumRadicalElectrons']]
    # Function to calculate descriptors
    def compute_descriptors(smiles, descriptor_functions):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [None] * len(descriptor_functions)
        return [func(mol) if mol else None for func in descriptor_functions]
    # Map descriptor names to functions
    descriptors_functions = [getattr(Descriptors, name) for name in descriptor_list]
    # Compute descriptors for each molecule
    descriptors_data = [compute_descriptors(sm, descriptors_functions) for sm in smiles_list]
    # Create DataFrame of descriptors
    descriptors_df = pd.DataFrame(descriptors_data, columns=descriptor_list)
    # what columns have all rows as zero 
    zero_cols = descriptors_df.columns[(descriptors_df == 0).all()]
    # remove columns with all zeros
    descriptors_df = descriptors_df.drop(zero_cols, axis=1)
    # scale the data
    scaler = StandardScaler()
    descriptors_df = pd.DataFrame(scaler.fit_transform(descriptors_df), columns=descriptors_df.columns)
    # fill NaN values with 0
    descriptors_df.fillna(0, inplace=True)
    return descriptors_df

def prepare_pca_data(original_df, generated_df):
    # columns that are common in both dataframes
    common_columns = list(set(original_df.columns).intersection(generated_df.columns))
    # keep only the common columns
    original_df = original_df[common_columns]
    generated_df = generated_df[common_columns]
    # arrange the columns in the same order
    generated_df = generated_df[original_df.columns]
    # PCA on the original data and generated data
    pca = PCA(n_components=2)
    original_data_pca = pca.fit_transform(original_df)
    generated_data_pca = pca.transform(generated_df)
    return original_data_pca, generated_data_pca


model_types = ['VAE', 'GAN', 'Transformer']
menu = st.sidebar.selectbox('Select model type', model_types)

if menu == 'VAE':
    st.title('Variational Autoencoder')
    st.markdown('''VAE is a generative model that learns to encode and decode data. It is used to generate new data similar to the training data.
                <br> VAE generates new molecules by learning and decoding essential patterns from training data.''', unsafe_allow_html=True)
    
    tab1, tabe2 = st.tabs(['Original Dataset', 'Generated molecules using saved models'])

    with tab1:
        with st.expander('Randomly select from original dataset'):
            num = int(st.number_input('Number of molecules', key = 'original', min_value=1, max_value=len(original_data_input_smiles_list)))
            selected_original_data = random.sample(original_data_input_smiles_list, num)
            # display the smiles and images of the selected molecules
            st.markdown('**Selected Molecules**')
            for i, sm in enumerate(selected_original_data):
                mol = Chem.MolFromSmiles(sm)
                if mol:
                    st.write(f'Molecule {i+1}')
                    st.write(f'SMILES: {sm}')
                    st.write('Molecular Structure:')
                    img = Draw.MolToImage(mol)
                    st.image(img)
                else:    
                    st.write(f"Failed to generate molecule from SMILES: {sm}")
    with tabe2:
        with st.expander('🟢 Model SurfMOL03212004'):
            n = int(st.number_input('Number of molecules'))
            button = st.button('Generate')
            if button:
                if n > 0:
                    st.warning('Generating molecules...')
        
                    encoded_flattened_dataset, height, vocab_itos, width = encode_data_from_smiles_to_vae_encoded(original_data_input_smiles_list)
                    selected_n_molecules = generate_vae_molecules(n, original_data_input_smiles_list, height, vocab_itos, width)

                    # Compute molecular descriptors for the generated molecules
                    generated_molecules_df = features_df(selected_n_molecules)
                    # Compute molecular descriptors for the original dataset
                    original_molecules_df = features_df(original_data_input_smiles_list)
                    # Prepare the data for PCA
                    original_data_pca, generated_data_pca = prepare_pca_data(original_molecules_df, generated_molecules_df)
                    
                    
                    df_original_data_pca = pd.DataFrame({
                    'Latent Dimension 1': original_data_pca[:, 0],
                    'Latent Dimension 2': original_data_pca[:, 1],
                    })

                    df_generated_data_pca  = pd.DataFrame({
                    'Latent Dimension 1': generated_data_pca [:, 0],
                    'Latent Dimension 2': generated_data_pca [:, 1],
                    })

                    # Create the initial scatter plot for the flattened dataset
                    fig = px.scatter(df_original_data_pca, 
                                    x='Latent Dimension 1', 
                                    y='Latent Dimension 2', 
                                    title='Latent Space Visualization',
                                    color_discrete_sequence=['#1f77b4'])  # Light blue color

                    # Add the scatter plot for the generated molecules
                    fig.add_trace(go.Scatter(x=df_generated_data_pca ['Latent Dimension 1'], 
                                            y=df_generated_data_pca ['Latent Dimension 2'], 
                                            mode='markers', 
                                            name='Generated Molecules',
                                            marker=dict(color='green')))  # Green color

                    # Update layout to include gridlines
                    fig.update_layout(
                        xaxis_title='Latent Dimension 1',
                        yaxis_title='Latent Dimension 2',
                        xaxis=dict(showgrid=True),  # Enable vertical gridlines
                        yaxis=dict(showgrid=True)   # Enable horizontal gridlines
                    )

                    # Display the plot in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

                    # Draw the selected molecules using RDKit along with their SMILES and display them in Streamlit
                    st.markdown('**Generated Molecules**')
                    for i, sm in enumerate(selected_n_molecules):
                        mol = Chem.MolFromSmiles(sm)
                        if mol:  # Check if the molecule was successfully created
                            st.write(f'Molecule {i+1}')
                            st.write(f'SMILES: {sm}')
                            st.write('Molecular Structure:')
                            # Convert the molecule to an image
                            img = Draw.MolToImage(mol)
                            # Display the image in Streamlit
                            st.image(img)
                        else:
                            st.write(f"Failed to generate molecule from SMILES: {sm}")
                else:
                    st.error('Please enter a valid number of molecules.')


elif menu == 'GAN':
    st.title('Generative Adversarial Network')
    st.markdown('''GAN is a generative model that creates new data by training two neural networks: a generator that produces data and a discriminator that evaluates it. <br>
              GANs can generate new molecules by learning to mimic the patterns found in molecular training data. ''', unsafe_allow_html=True)
    ### Add an image
    st.markdown("<h5 style='color:red'>Not yet deployed</h1>", unsafe_allow_html=True)
else:
    st.title('Transformers')
    st.markdown(''' A Transformer is a generative model that creates new data by training a neural network with attention mechanisms. <br>
             Transformers can generate new molecules by focusing on the intricate relationships within molecular structures found in the training data.''', unsafe_allow_html=True)
    ### Add an image
    st.markdown("<h5 style='color:red'>Not yet deployed</h1>", unsafe_allow_html=True)